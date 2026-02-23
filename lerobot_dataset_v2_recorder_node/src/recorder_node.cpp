/*
 * Copyright (c) 2026
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/image_transport.hpp>
#include <opencv2/opencv.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <vla_msgs/srv/save_record.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <std_srvs/srv/trigger.hpp>

// Apache Arrow / Parquet
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>
#include <parquet/properties.h>

// nlohmann JSON
#include <nlohmann/json.hpp>

// Standard
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>
#include <map>

namespace fs = std::filesystem;
using json = nlohmann::json;

enum class RecordState { IDLE, RECORDING, FINALIZING };

class LeRobotRecorderNode : public rclcpp::Node
{
public:
  explicit LeRobotRecorderNode(const rclcpp::NodeOptions & options)
  : Node("lerobot_recorder", options)
  {
    // ── Parameters ───────────────────────────────────────────────────────────
    this->declare_parameter("dataset_root", "/tmp/lerobot_dataset");
    this->declare_parameter("image_topics", std::vector<std::string>{});
    this->declare_parameter("camera_names", std::vector<std::string>{});
    this->declare_parameter("record_rate_hz", 30.0);
    this->declare_parameter("video_fps", 30.0);
    this->declare_parameter("image_encoding", std::string("bgr8"));
    this->declare_parameter("chunk_index", 0);
    this->declare_parameter("drop_policy", std::string("skip"));
    this->declare_parameter("max_episode_seconds", 0.0);
    this->declare_parameter("joint_state_topic", std::string("/joint_states"));
    this->declare_parameter("follower_joint_state_topic", std::string("/follower/joint_states"));
    // Staleness threshold: image older than this is treated as stale.
    // Default 1.0s accommodates cameras running well below record_rate_hz (e.g. 5 Hz).
    this->declare_parameter("image_stale_timeout_sec", 1.0);

    dataset_root_ = this->get_parameter("dataset_root").as_string();
    image_topics_ = this->get_parameter("image_topics").as_string_array();
    camera_names_ = this->get_parameter("camera_names").as_string_array();
    record_rate_hz_ = this->get_parameter("record_rate_hz").as_double();
    video_fps_ = this->get_parameter("video_fps").as_double();
    image_encoding_ = this->get_parameter("image_encoding").as_string();
    chunk_index_ = this->get_parameter("chunk_index").as_int();
    drop_policy_ = this->get_parameter("drop_policy").as_string();
    max_episode_seconds_ = this->get_parameter("max_episode_seconds").as_double();
    joint_state_topic_ = this->get_parameter("joint_state_topic").as_string();
    follower_joint_state_topic_ = this->get_parameter("follower_joint_state_topic").as_string();
    image_stale_timeout_sec_ = this->get_parameter("image_stale_timeout_sec").as_double();

    if (image_topics_.size() != camera_names_.size()) {
      RCLCPP_ERROR(
        get_logger(),
        "image_topics (%zu) and camera_names (%zu) must have the same length",
        image_topics_.size(), camera_names_.size());
      throw std::runtime_error("Parameter mismatch: image_topics / camera_names");
    }

    const size_t n_cams = image_topics_.size();
    latest_images_.resize(n_cams);
    latest_image_times_.resize(n_cams, rclcpp::Time(0, 0, RCL_ROS_TIME));
    actual_image_sizes_.assign(n_cams, cv::Size(0, 0));

    // ── Directory structure ──────────────────────────────────────────────────
    ensureDirectories();

    // ── Load existing meta ───────────────────────────────────────────────────
    loadExistingTasks();
    episode_index_ = getNextEpisodeIndex();
    initInfoJson();

    // ── Subscribers ──────────────────────────────────────────────────────────
    for (size_t i = 0; i < n_cams; i++) {
      // Use sensor_data QoS (BEST_EFFORT) so we can receive from both
      // RELIABLE and BEST_EFFORT publishers (e.g. usb_cam, v4l2_camera)
      auto sub = image_transport::create_subscription(
        this, image_topics_[i],
        [this, i](const sensor_msgs::msg::Image::ConstSharedPtr & msg) {
          imageCallback(i, msg);
        },
        "raw",
        rmw_qos_profile_sensor_data);
      image_subs_.push_back(sub);
      RCLCPP_INFO(
        get_logger(), "Subscribed image[%zu]: %s -> camera '%s' (QoS: sensor_data)",
        i, image_topics_[i].c_str(), camera_names_[i].c_str());
    }

    // SensorDataQoS = BEST_EFFORT + VOLATILE; compatible with both RELIABLE and BEST_EFFORT publishers
    joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      joint_state_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&LeRobotRecorderNode::jointStateCallback, this, std::placeholders::_1));
    RCLCPP_INFO(
      get_logger(), "Subscribed joint states: %s (QoS: sensor_data)",
      joint_state_topic_.c_str());

    follower_joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      follower_joint_state_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&LeRobotRecorderNode::followerJointStateCallback, this, std::placeholders::_1));
    RCLCPP_INFO(
      get_logger(), "Subscribed follower joint states: %s (QoS: sensor_data)",
      follower_joint_state_topic_.c_str());

    // ── Services ─────────────────────────────────────────────────────────────
    start_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "start_record",
      std::bind(
        &LeRobotRecorderNode::startCallback, this,
        std::placeholders::_1, std::placeholders::_2));
    stop_srv_ = this->create_service<vla_msgs::srv::SaveRecord>(
      "stop_record",
      std::bind(
        &LeRobotRecorderNode::stopCallback, this,
        std::placeholders::_1, std::placeholders::_2));
    RCLCPP_INFO(
      get_logger(),
      "Services ready: ~/start_record (Trigger), ~/stop_record (SaveRecord with prompt)");

    // ── Timer ─────────────────────────────────────────────────────────────
    auto period = std::chrono::duration<double>(1.0 / record_rate_hz_);
    timer_ = this->create_wall_timer(
      period, std::bind(&LeRobotRecorderNode::timerCallback, this));

    RCLCPP_INFO(
      get_logger(),
      "LeRobot recorder ready. dataset_root='%s', next_episode=%d, chunk=%d",
      dataset_root_.c_str(), episode_index_, chunk_index_);
  }

private:
  // ── Parameters ────────────────────────────────────────────────────────────
  std::string dataset_root_;
  std::vector<std::string> image_topics_;
  std::vector<std::string> camera_names_;
  double record_rate_hz_{30.0};
  double video_fps_{30.0};
  std::string image_encoding_{"bgr8"};
  int chunk_index_{0};
  std::string drop_policy_{"skip"};
  double max_episode_seconds_{0.0};
  std::string joint_state_topic_{"/joint_states"};
  std::string follower_joint_state_topic_{"/follower/joint_states"};
  double image_stale_timeout_sec_{1.0};

  // ── State ─────────────────────────────────────────────────────────────────
  RecordState state_{RecordState::IDLE};
  std::mutex mutex_;

  // ── ROS interfaces ────────────────────────────────────────────────────────
  std::vector<image_transport::Subscriber> image_subs_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr follower_joint_state_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr start_srv_;
  rclcpp::Service<vla_msgs::srv::SaveRecord>::SharedPtr stop_srv_;
  rclcpp::TimerBase::SharedPtr timer_;

  // ── Latest data (mutex-protected) ─────────────────────────────────────────
  std::vector<cv::Mat> latest_images_;
  std::vector<rclcpp::Time> latest_image_times_;
  sensor_msgs::msg::JointState latest_joint_state_;
  bool joint_state_received_{false};
  sensor_msgs::msg::JointState latest_follower_joint_state_;
  bool follower_joint_state_received_{false};

  // ── Actual image sizes (updated once per camera, persists across episodes) ─
  std::vector<cv::Size> actual_image_sizes_;

  // ── Actual joint state dimensions (updated on first message received) ─────
  int actual_state_dim_{-1};
  int actual_follower_state_dim_{-1};
  std::vector<std::string> joint_state_names_;
  std::vector<std::string> follower_joint_state_names_;

  // ── Episode state ─────────────────────────────────────────────────────────
  int episode_index_{0};
  int task_index_{-1};
  int frame_index_{0};
  rclcpp::Time episode_start_time_;
  std::string current_prompt_;
  double ts_{0.0};   // camera-capture-based timestamp for current frame (set in timerCallback)
  // Reference camera time of the first recorded frame (ts=0 anchor)
  rclcpp::Time first_cam_ts_;
  bool first_cam_ts_set_{false};
  // Camera timestamps used for the last recorded frame (duplicate detection)
  std::vector<rclcpp::Time> last_recorded_image_times_;

  // ── Video writers ─────────────────────────────────────────────────────────
  std::vector<std::string> video_paths_;
  std::vector<cv::VideoWriter> video_writers_;
  std::vector<cv::Size> expected_video_sizes_;
  std::vector<bool> video_size_determined_;

  // ── Frame data buffers ────────────────────────────────────────────────────
  std::vector<double> timestamps_;
  std::vector<int64_t> frame_indices_;
  std::vector<int64_t> task_indices_data_;
  std::vector<std::vector<float>> states_;
  std::vector<std::vector<float>> follower_states_;

  // ── Meta ──────────────────────────────────────────────────────────────────
  std::map<std::string, int> task_map_;   // prompt -> task_index
  int next_task_index_{0};

  // ── Helpers ───────────────────────────────────────────────────────────────
  void ensureDirectories()
  {
    auto mkdir = [this](const std::string & path) {
        std::error_code ec;
        fs::create_directories(path, ec);
        if (ec) {
          RCLCPP_ERROR(
            get_logger(), "Failed to create directory '%s': %s",
            path.c_str(), ec.message().c_str());
        } else {
          RCLCPP_DEBUG(get_logger(), "Directory ready: %s", path.c_str());
        }
      };

    mkdir(dataset_root_ + "/data/chunk-" + chunkStr());
    mkdir(dataset_root_ + "/meta");
    for (const auto & cam : camera_names_) {
      mkdir(dataset_root_ + "/videos/chunk-" + chunkStr() + "/" + cam);
    }
  }

  std::string chunkStr() const
  {
    char buf[8]; snprintf(buf, sizeof(buf), "%03d", chunk_index_);
    return std::string(buf);
  }

  std::string episodeStr(int ep) const
  {
    char buf[10]; snprintf(buf, sizeof(buf), "%06d", ep);
    return std::string(buf);
  }

  // ── Callbacks ─────────────────────────────────────────────────────────────
  void imageCallback(
    size_t idx,
    const sensor_msgs::msg::Image::ConstSharedPtr & msg)
  {
    try {
      auto cv_ptr = cv_bridge::toCvShare(msg, image_encoding_);
      std::lock_guard<std::mutex> lock(mutex_);
      latest_images_[idx] = cv_ptr->image.clone();
      latest_image_times_[idx] = msg->header.stamp;
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "cv_bridge exception (cam %zu): %s", idx, e.what());
    }
  }

  void jointStateCallback(const sensor_msgs::msg::JointState::ConstSharedPtr & msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_joint_state_ = *msg;
    joint_state_received_ = true;
    // Capture dimension and joint names on first message
    if (actual_state_dim_ < 0 && !msg->position.empty()) {
      actual_state_dim_ = static_cast<int>(msg->position.size());
      joint_state_names_ = msg->name;   // may be empty for some publishers
    }
  }

  void followerJointStateCallback(const sensor_msgs::msg::JointState::ConstSharedPtr & msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_follower_joint_state_ = *msg;
    follower_joint_state_received_ = true;
    if (actual_follower_state_dim_ < 0 && !msg->position.empty()) {
      actual_follower_state_dim_ = static_cast<int>(msg->position.size());
      follower_joint_state_names_ = msg->name;
    }
  }

  void timerCallback()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != RecordState::RECORDING) {return;}

    // Max episode duration guard
    if (max_episode_seconds_ > 0.0) {
      double elapsed = (get_clock()->now() - episode_start_time_).seconds();
      if (elapsed > max_episode_seconds_) {
        RCLCPP_WARN(
          get_logger(),
          "Max episode duration %.1fs reached. Auto-stopping.", max_episode_seconds_);
        finalizeEpisodeUnlocked();
        return;
      }
    }

    auto now_ros = get_clock()->now();
    double stale_sec = image_stale_timeout_sec_;

    // ── Check joint states ───────────────────────────────────────────────
    if (!joint_state_received_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "no joint_state received on '%s' (policy=%s) — frame skipped",
        joint_state_topic_.c_str(), drop_policy_.c_str());
      handleMissingData("joint_state", drop_policy_);
      return;
    }
    if (!follower_joint_state_received_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "no follower_joint_state received on '%s' (policy=%s) — frame skipped",
        follower_joint_state_topic_.c_str(), drop_policy_.c_str());
      handleMissingData("follower_joint_state", drop_policy_);
      return;
    }

    // ── Pass 1: validate ALL cameras before writing anything ─────────────
    // (must check all cameras first; only if all OK do we write video+Parquet)
    bool should_skip = false;
    for (size_t i = 0; i < camera_names_.size(); i++) {
      if (latest_images_[i].empty()) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "no image received for camera=%s (policy=%s)",
          camera_names_[i].c_str(), drop_policy_.c_str());
        if (drop_policy_ == "stop") {
          finalizeEpisodeUnlocked();
          return;
        }
        if (drop_policy_ == "skip") {
          should_skip = true;
        }
        continue;
      }

      // Staleness check
      if (latest_image_times_[i].nanoseconds() > 0) {
        double age = (now_ros - latest_image_times_[i]).seconds();
        if (age > stale_sec) {
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 1000,
            "stale image for camera=%s (age=%.3fs, policy=%s)",
            camera_names_[i].c_str(), age, drop_policy_.c_str());
          if (drop_policy_ == "stop") {
            finalizeEpisodeUnlocked();
            return;
          }
          if (drop_policy_ == "skip") {
            should_skip = true;
          }
        }
      }

      // Size consistency check (against already-determined video size)
      if (video_size_determined_[i] &&
        latest_images_[i].size() != expected_video_sizes_[i])
      {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "frame size mismatch for camera=%s, skipping frame",
          camera_names_[i].c_str());
        should_skip = true;
      }
    }

    // If any camera is not ready, skip this entire frame (no video, no Parquet)
    if (should_skip) {return;}

    // ── Duplicate-frame guard: skip if no camera has a new image ─────────
    // When cameras publish slower than the timer (e.g. 5Hz vs 30Hz), the same
    // image would be written multiple times, creating duplicate timestamps.
    // Only proceed if ALL cameras have advanced their image since the last write.
    {
      bool all_cameras_new = true;
      for (size_t i = 0; i < camera_names_.size(); i++) {
        if (latest_image_times_[i] == last_recorded_image_times_[i]) {
          all_cameras_new = false;
          break;
        }
      }
      if (!all_cameras_new) {return;}
    }

    // ── Compute timestamp anchored to the first recorded camera frame ─────
    // avg of all camera header timestamps = actual capture time of this frame.
    // ts=0 is defined as the first recorded frame (avoids negative ts when
    // cameras were already publishing before startRecording was called).
    {
      int64_t ns_sum = 0;
      int valid = 0;
      for (size_t i = 0; i < camera_names_.size(); i++) {
        if (latest_image_times_[i].nanoseconds() > 0) {
          ns_sum += latest_image_times_[i].nanoseconds();
          valid++;
        }
      }
      rclcpp::Time cam_now = (valid > 0) ?
        rclcpp::Time(ns_sum / valid, RCL_ROS_TIME) :
        now_ros;
      if (!first_cam_ts_set_) {
        first_cam_ts_ = cam_now;
        first_cam_ts_set_ = true;
      }
      ts_ = (cam_now - first_cam_ts_).seconds();
    }

    // ── Pass 2: all cameras OK & new — write video frames then Parquet ───
    for (size_t i = 0; i < camera_names_.size(); i++) {
      // Lazy video writer init on first valid frame
      if (!video_size_determined_[i]) {
        cv::Size vsz = latest_images_[i].size();
        expected_video_sizes_[i] = vsz;
        video_size_determined_[i] = true;
        actual_image_sizes_[i] = vsz;   // persist across episodes for info.json

        int fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
        video_writers_[i].open(video_paths_[i], fourcc, video_fps_, vsz, true);
        if (!video_writers_[i].isOpened()) {
          RCLCPP_WARN(
            get_logger(),
            "avc1 failed, falling back to mp4v for camera=%s",
            camera_names_[i].c_str());
          fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
          video_writers_[i].open(video_paths_[i], fourcc, video_fps_, vsz, true);
        }
        if (!video_writers_[i].isOpened()) {
          RCLCPP_ERROR(
            get_logger(),
            "video writer open failed (fourcc=avc1, fallback=mp4v also failed) "
            "for camera=%s path=%s",
            camera_names_[i].c_str(), video_paths_[i].c_str());
        } else {
          RCLCPP_INFO(
            get_logger(), "Video writer opened: %s (%dx%d @ %.1ffps)",
            video_paths_[i].c_str(), vsz.width, vsz.height, video_fps_);
        }
      }

      if (video_writers_[i].isOpened()) {
        video_writers_[i].write(latest_images_[i]);
      }
      // Mark this camera's timestamp as "recorded" for duplicate detection
      last_recorded_image_times_[i] = latest_image_times_[i];
    }

    // ── Record Parquet row ────────────────────────────────────────────────
    timestamps_.push_back(ts_);
    frame_indices_.push_back(frame_index_);
    task_indices_data_.push_back(task_index_);

    std::vector<float> state;
    state.reserve(latest_joint_state_.position.size());
    for (double p : latest_joint_state_.position) {
      state.push_back(static_cast<float>(p));
    }
    states_.push_back(std::move(state));

    std::vector<float> follower_state;
    follower_state.reserve(latest_follower_joint_state_.position.size());
    for (double p : latest_follower_joint_state_.position) {
      follower_state.push_back(static_cast<float>(p));
    }
    follower_states_.push_back(std::move(follower_state));

    frame_index_++;
  }

  // Returns false if frame should be skipped; may finalize if policy=stop
  bool handleMissingData(const std::string & source, const std::string & policy)
  {
    if (policy == "stop") {
      RCLCPP_ERROR(get_logger(), "Missing data: %s. Stopping.", source.c_str());
      finalizeEpisodeUnlocked();
      return false;
    }
    if (policy == "skip") {
      return false;
    }
    // duplicate: caller continues with stale data
    return true;
  }

  // ── Start service handler (start_record) ──────────────────────────────────
  void startCallback(
    const std_srvs::srv::Trigger::Request::SharedPtr /*req*/,
    std_srvs::srv::Trigger::Response::SharedPtr resp)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (state_ == RecordState::FINALIZING) {
      resp->success = false;
      resp->message = "finalizing in progress";
      return;
    }
    if (state_ == RecordState::RECORDING) {
      resp->success = false;
      resp->message = "already recording";
      return;
    }
    // IDLE -> RECORDING
    int ep = episode_index_;
    startRecordingUnlocked();
    resp->success = true;
    resp->message = "started episode " + std::to_string(ep) +
      " (prompt will be provided at stop)";
    RCLCPP_INFO(get_logger(), "Recording started: episode=%d", ep);
  }

  // ── Stop service handler (stop_record) ───────────────────────────────────
  void stopCallback(
    const vla_msgs::srv::SaveRecord::Request::SharedPtr req,
    vla_msgs::srv::SaveRecord::Response::SharedPtr resp)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (state_ == RecordState::IDLE) {
      // Already stopped: succeed silently
      resp->success = true;
      resp->message = "not recording (already stopped)";
      return;
    }
    if (state_ == RecordState::FINALIZING) {
      resp->success = false;
      resp->message = "finalizing in progress";
      return;
    }
    // RECORDING -> FINALIZING -> IDLE
    // Set task prompt now (deferred from start to stop)
    current_prompt_ = req->prompt;
    task_index_ = getOrCreateTaskIndex(req->prompt);
    // Backfill task_index for all frames buffered during this episode
    std::fill(
      task_indices_data_.begin(), task_indices_data_.end(),
      static_cast<int64_t>(task_index_));

    double duration = (get_clock()->now() - episode_start_time_).seconds();
    int completed_frames = frame_index_;
    finalizeEpisodeUnlocked();
    resp->success = true;
    resp->message = "episode finalized: " + std::to_string(completed_frames) +
      " frames in " + std::to_string(duration) + "s, prompt='" + req->prompt + "'";
    RCLCPP_INFO(
      get_logger(), "Recording stopped: episode=%d task='%s'",
      episode_index_ - 1, req->prompt.c_str());
  }

  // ── Recording control (called with mutex held) ────────────────────────────
  void startRecordingUnlocked()
  {
    current_prompt_ = "";
    task_index_ = -1;   // resolved at stop time via SaveRecord prompt
    frame_index_ = 0;
    episode_start_time_ = get_clock()->now();

    // Reset timestamp anchor and duplicate-detection state
    first_cam_ts_set_ = false;
    first_cam_ts_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    last_recorded_image_times_.assign(
      camera_names_.size(),
      rclcpp::Time(0, 0, RCL_ROS_TIME));

    // Clear buffers
    timestamps_.clear();
    frame_indices_.clear();
    task_indices_data_.clear();
    states_.clear();
    follower_states_.clear();

    // Prepare video writers (lazily opened on first frame)
    const size_t n = camera_names_.size();
    video_paths_.resize(n);
    video_writers_.resize(n);
    expected_video_sizes_.assign(n, cv::Size(0, 0));
    video_size_determined_.assign(n, false);

    for (size_t i = 0; i < n; i++) {
      video_paths_[i] =
        dataset_root_ + "/videos/chunk-" + chunkStr() +
        "/" + camera_names_[i] +
        "/episode_" + episodeStr(episode_index_) + ".mp4";
      fs::create_directories(fs::path(video_paths_[i]).parent_path());
    }

    // Ensure data directory exists (in case dataset_root was created externally after startup)
    fs::create_directories(dataset_root_ + "/data/chunk-" + chunkStr());

    state_ = RecordState::RECORDING;
  }

  void finalizeEpisodeUnlocked()
  {
    state_ = RecordState::FINALIZING;
    int ep = episode_index_;
    int nf = frame_index_;
    RCLCPP_INFO(get_logger(), "Finalizing episode %d (%d frames)...", ep, nf);

    // Resolve task_index if auto-stopped (timer) without a stop_record call
    if (task_index_ < 0) {
      task_index_ = getOrCreateTaskIndex("");
      RCLCPP_WARN(
        get_logger(),
        "Episode %d auto-finalized without prompt; task saved as empty string.", ep);
      std::fill(
        task_indices_data_.begin(), task_indices_data_.end(),
        static_cast<int64_t>(task_index_));
    }

    // Close video writers
    for (size_t i = 0; i < video_writers_.size(); i++) {
      if (video_writers_[i].isOpened()) {
        video_writers_[i].release();
        RCLCPP_INFO(get_logger(), "Video closed: camera=%s", camera_names_[i].c_str());
      }
    }
    video_writers_.clear();

    // Write Parquet
    if (nf > 0) {
      std::string pq_path =
        dataset_root_ + "/data/chunk-" + chunkStr() +
        "/episode_" + episodeStr(ep) + ".parquet";
      auto st = writeParquetFile(pq_path);
      if (!st.ok()) {
        RCLCPP_ERROR(get_logger(), "Parquet write failed: %s", st.ToString().c_str());
      } else {
        RCLCPP_INFO(get_logger(), "Parquet written: %s", pq_path.c_str());
      }
    }

    // Update meta
    appendEpisodeMeta(ep, task_index_, nf);
    updateInfoJson();

    episode_index_++;
    state_ = RecordState::IDLE;
    RCLCPP_INFO(get_logger(), "Episode %d finalized. Next episode: %d", ep, episode_index_);
  }

  // ── Task / meta helpers ───────────────────────────────────────────────────
  int getOrCreateTaskIndex(const std::string & prompt)
  {
    auto it = task_map_.find(prompt);
    if (it != task_map_.end()) {return it->second;}
    int idx = next_task_index_++;
    task_map_[prompt] = idx;
    appendTaskMeta(idx, prompt);
    return idx;
  }

  void appendTaskMeta(int idx, const std::string & prompt)
  {
    std::ofstream f(dataset_root_ + "/meta/tasks.jsonl", std::ios::app);
    json entry;
    entry["task_index"] = idx;
    entry["task"] = prompt;
    f << entry.dump() << "\n";
  }

  void appendEpisodeMeta(int ep_idx, int task_idx, int num_frames)
  {
    std::ofstream f(dataset_root_ + "/meta/episodes.jsonl", std::ios::app);
    json entry;
    entry["episode_index"] = ep_idx;
    entry["task_index"] = task_idx;
    entry["num_frames"] = num_frames;
    entry["chunk"] = chunk_index_;
    f << entry.dump() << "\n";
  }

  void loadExistingTasks()
  {
    std::string path = dataset_root_ + "/meta/tasks.jsonl";
    if (!fs::exists(path)) {return;}
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
      if (line.empty()) {continue;}
      try {
        auto j = json::parse(line);
        int idx = j.at("task_index").get<int>();
        std::string t = j.at("task").get<std::string>();
        task_map_[t] = idx;
        next_task_index_ = std::max(next_task_index_, idx + 1);
      } catch (...) {}
    }
    RCLCPP_INFO(get_logger(), "Loaded %zu existing tasks", task_map_.size());
  }

  int getNextEpisodeIndex()
  {
    std::string data_dir = dataset_root_ + "/data/chunk-" + chunkStr();
    if (!fs::exists(data_dir)) {return 0;}
    int max_idx = -1;
    for (const auto & e : fs::directory_iterator(data_dir)) {
      std::string name = e.path().filename().string();
      // episode_NNNNNN.parquet
      if (name.size() >= 22 &&
        name.substr(0, 8) == "episode_" &&
        name.substr(name.size() - 8) == ".parquet")
      {
        try {
          int idx = std::stoi(name.substr(8, 6));
          max_idx = std::max(max_idx, idx);
        } catch (...) {}
      }
    }
    return max_idx + 1;
  }

  void initInfoJson()
  {
    std::string info_path = dataset_root_ + "/meta/info.json";
    if (fs::exists(info_path)) {return;}

    json info = getDefaultInfo();

    // Try to load template from package share
    try {
      std::string pkg_dir =
        ament_index_cpp::get_package_share_directory("lerobot_dataset_v2_recorder_node");
      std::string tmpl = pkg_dir + "/resources/info_template.json";
      if (fs::exists(tmpl)) {
        std::ifstream tf(tmpl);
        tf >> info;
        RCLCPP_INFO(get_logger(), "Loaded info template: %s", tmpl.c_str());
      }
    } catch (...) {}

    applyRuntimeInfo(info);
    std::ofstream of(info_path);
    of << info.dump(2) << "\n";
    RCLCPP_INFO(get_logger(), "Created meta/info.json");
  }

  void updateInfoJson()
  {
    std::string info_path = dataset_root_ + "/meta/info.json";
    json info = getDefaultInfo();
    if (fs::exists(info_path)) {
      try {
        std::ifstream f(info_path);
        f >> info;
      } catch (...) {}
    }

    // Re-apply keys (guard against stale info.json from older runs)
    {
      json keys = json::array();
      keys.push_back("observation.state");
      keys.push_back("observation.follower_state");
      info["keys"] = keys;
    }

    // Recount total_episodes and total_frames from episodes.jsonl
    // (episode_index_ is NOT yet incremented when this is called from finalizeEpisodeUnlocked)
    int total_episodes = 0;
    int total_frames = 0;
    std::string episodes_path = dataset_root_ + "/meta/episodes.jsonl";
    if (fs::exists(episodes_path)) {
      std::ifstream f(episodes_path);
      std::string line;
      while (std::getline(f, line)) {
        if (line.empty()) {continue;}
        try {
          auto j = json::parse(line);
          total_frames += j.at("num_frames").get<int>();
          total_episodes++;
        } catch (...) {}
      }
    }

    info["total_episodes"] = total_episodes;
    info["total_frames"] = total_frames;
    info["total_tasks"] = static_cast<int>(task_map_.size());
    info["total_videos"] = total_episodes * static_cast<int>(camera_names_.size());

    // Update actual image shapes when known
    for (size_t i = 0; i < camera_names_.size(); i++) {
      if (actual_image_sizes_[i].width > 0 && actual_image_sizes_[i].height > 0) {
        info["shapes"]["observation.images." + camera_names_[i]] =
          json::array({3, actual_image_sizes_[i].height, actual_image_sizes_[i].width});
      }
    }

    // Update observation.state / observation.follower_state shapes from actual joint counts
    if (actual_state_dim_ > 0) {
      info["shapes"]["observation.state"] = json::array({actual_state_dim_});
    }
    if (actual_follower_state_dim_ > 0) {
      info["shapes"]["observation.follower_state"] = json::array({actual_follower_state_dim_});
    }

    // Populate names with joint names if available
    if (!joint_state_names_.empty()) {
      info["names"]["observation.state"] = joint_state_names_;
    }
    if (!follower_joint_state_names_.empty()) {
      info["names"]["observation.follower_state"] = follower_joint_state_names_;
    }

    // Refresh splits
    info["splits"] = json::object();
    info["splits"]["train"] = "0:" + std::to_string(total_episodes);

    std::ofstream of(info_path);
    of << info.dump(2) << "\n";
  }

  json getDefaultInfo()
  {
    json info;
    info["codebase_version"] = "v2.1";
    info["fps"] = video_fps_;
    info["robot_type"] = "unknown";
    info["keys"] = json::array({"observation.state", "observation.follower_state"});
    info["video_keys"] = json::array();
    info["shapes"] = json::object();
    info["names"] = json::object();
    info["total_episodes"] = 0;
    info["total_frames"] = 0;
    info["total_tasks"] = 0;
    info["total_videos"] = 0;
    info["total_chunks"] = 1;
    info["chunks_size"] = 1000;
    info["data_path"] = "data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet";
    info["video_path"] = "videos/chunk-{chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4";
    info["splits"] = json::object({{"train", "0:0"}});
    info["license"] = "apache-2.0";
    info["tags"] = json::array();
    return info;
  }

  void applyRuntimeInfo(json & info)
  {
    info["fps"] = video_fps_;

    json video_keys = json::array();
    for (const auto & cam : camera_names_) {
      video_keys.push_back("observation.images." + cam);
      info["shapes"]["observation.images." + cam] = json::array({3, -1, -1});
    }
    info["video_keys"] = video_keys;
    info["data_path"] = "data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet";
    info["video_path"] = "videos/chunk-{chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4";
  }

  // ── Parquet writer ────────────────────────────────────────────────────────
  arrow::Status writeParquetFile(const std::string & path)
  {
    if (timestamps_.empty()) {
      return arrow::Status::OK();
    }

    // timestamp column
    arrow::DoubleBuilder ts_builder;
    ARROW_RETURN_NOT_OK(ts_builder.AppendValues(timestamps_));
    std::shared_ptr<arrow::Array> ts_array;
    ARROW_RETURN_NOT_OK(ts_builder.Finish(&ts_array));

    // frame_index column
    arrow::Int64Builder fi_builder;
    ARROW_RETURN_NOT_OK(fi_builder.AppendValues(frame_indices_));
    std::shared_ptr<arrow::Array> fi_array;
    ARROW_RETURN_NOT_OK(fi_builder.Finish(&fi_array));

    // task_index column
    arrow::Int64Builder ti_builder;
    ARROW_RETURN_NOT_OK(ti_builder.AppendValues(task_indices_data_));
    std::shared_ptr<arrow::Array> ti_array;
    ARROW_RETURN_NOT_OK(ti_builder.Finish(&ti_array));

    // observation.state column (list<float32>)
    auto state_float_builder = std::make_shared<arrow::FloatBuilder>();
    arrow::ListBuilder state_list_builder(
      arrow::default_memory_pool(), state_float_builder, arrow::list(arrow::float32()));

    for (const auto & state : states_) {
      ARROW_RETURN_NOT_OK(state_list_builder.Append());
      ARROW_RETURN_NOT_OK(
        state_float_builder->AppendValues(
          state.data(), static_cast<int64_t>(state.size())));
    }
    std::shared_ptr<arrow::Array> state_array;
    ARROW_RETURN_NOT_OK(state_list_builder.Finish(&state_array));

    // observation.follower_state column (list<float32>)
    auto follower_float_builder = std::make_shared<arrow::FloatBuilder>();
    arrow::ListBuilder follower_list_builder(
      arrow::default_memory_pool(), follower_float_builder, arrow::list(arrow::float32()));

    for (const auto & fs : follower_states_) {
      ARROW_RETURN_NOT_OK(follower_list_builder.Append());
      ARROW_RETURN_NOT_OK(
        follower_float_builder->AppendValues(
          fs.data(), static_cast<int64_t>(fs.size())));
    }
    std::shared_ptr<arrow::Array> follower_state_array;
    ARROW_RETURN_NOT_OK(follower_list_builder.Finish(&follower_state_array));

    // Schema & table
    auto schema = arrow::schema(
    {
      arrow::field("timestamp", arrow::float64()),
      arrow::field("frame_index", arrow::int64()),
      arrow::field("task_index", arrow::int64()),
      arrow::field("observation.state", arrow::list(arrow::float32())),
      arrow::field("observation.follower_state", arrow::list(arrow::float32())),
    });

    auto table = arrow::Table::Make(
      schema, {ts_array, fi_array, ti_array, state_array, follower_state_array});

    // Write
    std::shared_ptr<arrow::io::FileOutputStream> outfile;
    ARROW_ASSIGN_OR_RAISE(outfile, arrow::io::FileOutputStream::Open(path));

    auto props = parquet::WriterProperties::Builder()
      .compression(parquet::Compression::SNAPPY)
      ->build();
    auto arrow_props = parquet::ArrowWriterProperties::Builder().build();

    ARROW_RETURN_NOT_OK(
      parquet::arrow::WriteTable(
        *table,
        arrow::default_memory_pool(),
        std::static_pointer_cast<arrow::io::OutputStream>(outfile),
        static_cast<int64_t>(table->num_rows()),
        props,
        arrow_props));

    return arrow::Status::OK();
  }
};

RCLCPP_COMPONENTS_REGISTER_NODE(LeRobotRecorderNode)
