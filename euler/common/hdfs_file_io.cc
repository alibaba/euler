/* Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <stdlib.h>

#include <string>
#include <vector>
#include <functional>

#include "hdfs/hdfs.h"

#include "euler/common/file_io.h"
#include "euler/common/env.h"
#include "euler/common/logging.h"
#include "euler/common/str_util.h"

namespace euler {

const size_t kBufferSize = 4096;

template <typename R, typename... Args>
Status BindFunc(void* handle, const char* name,
                std::function<R(Args...)>* func) {
  void* symbol_ptr = nullptr;
  RETURN_IF_ERROR(
      Env::Default()->GetSymbolFromLibrary(handle, name, &symbol_ptr));
  *func = reinterpret_cast<R (*)(Args...)>(symbol_ptr);
  return Status::OK();
}

class LibHDFS {
 public:
  static LibHDFS* Load() {
    static LibHDFS* lib = []() -> LibHDFS* {
      LibHDFS* lib = new LibHDFS;
      lib->LoadAndBind();
      return lib;
    }();

    return lib;
  }

  Status status() { return status_; }

  std::function<hdfsFS(hdfsBuilder*)> hdfsBuilderConnect;
  std::function<hdfsBuilder*()> hdfsNewBuilder;
  std::function<void(hdfsBuilder*, const char*)> hdfsBuilderSetNameNode;
  std::function<void(hdfsBuilder*, tPort)> hdfsBuilderSetNameNodePort;
  std::function<int(const char*, char**)> hdfsConfGetStr;
  std::function<int(hdfsFS, hdfsFile)> hdfsCloseFile;
  std::function<
    tSize(hdfsFS fs, hdfsFile file, void* buffer, tSize length)>hdfsRead;
  std::function<tSize(hdfsFS, hdfsFile, tOffset, void*, tSize)> hdfsPread;
  std::function<tSize(hdfsFS, hdfsFile, const void*, tSize)> hdfsWrite;
  std::function<int(hdfsFS, hdfsFile)> hdfsHFlush;
  std::function<int(hdfsFS, hdfsFile)> hdfsHSync;
  std::function<tOffset(hdfsFS, hdfsFile)> hdfsTell;
  std::function<
    hdfsFile(hdfsFS, const char*, int, int, short, tSize)> hdfsOpenFile;
  std::function<int(hdfsFS, const char*)> hdfsExists;
  std::function<hdfsFileInfo*(hdfsFS, const char*, int*)> hdfsListDirectory;
  std::function<void(hdfsFileInfo*, int)> hdfsFreeFileInfo;
  std::function<int(hdfsFS, const char*, int recursive)> hdfsDelete;
  std::function<int(hdfsFS, const char*)> hdfsCreateDirectory;
  std::function<hdfsFileInfo*(hdfsFS, const char*)> hdfsGetPathInfo;
  std::function<int(hdfsFS, const char*, const char*)> hdfsRename;

 private:
  void LoadAndBind() {
    auto TryLoadAndBind = [this](const char* name, void** handle) -> Status {
      RETURN_IF_ERROR(Env::Default()->LoadLibrary(name, handle));
#define BIND_HDFS_FUNC(function) \
      RETURN_IF_ERROR(BindFunc(*handle, #function, &function));

      BIND_HDFS_FUNC(hdfsBuilderConnect);
      BIND_HDFS_FUNC(hdfsNewBuilder);
      BIND_HDFS_FUNC(hdfsBuilderSetNameNode);
      BIND_HDFS_FUNC(hdfsBuilderSetNameNodePort);
      BIND_HDFS_FUNC(hdfsConfGetStr);
      BIND_HDFS_FUNC(hdfsCloseFile);
      BIND_HDFS_FUNC(hdfsRead);
      BIND_HDFS_FUNC(hdfsPread);
      BIND_HDFS_FUNC(hdfsWrite);
      BIND_HDFS_FUNC(hdfsHFlush);
      BIND_HDFS_FUNC(hdfsTell);
      BIND_HDFS_FUNC(hdfsHSync);
      BIND_HDFS_FUNC(hdfsOpenFile);
      BIND_HDFS_FUNC(hdfsExists);
      BIND_HDFS_FUNC(hdfsListDirectory);
      BIND_HDFS_FUNC(hdfsFreeFileInfo);
      BIND_HDFS_FUNC(hdfsDelete);
      BIND_HDFS_FUNC(hdfsCreateDirectory);
      BIND_HDFS_FUNC(hdfsGetPathInfo);
      BIND_HDFS_FUNC(hdfsRename);
#undef BIND_HDFS_FUNC
      return Status::OK();
    };

    const char* kLibHdfsDso = "libhdfs.so";
    char* hdfs_home = getenv("HADOOP_HDFS_HOME");
    if (hdfs_home != nullptr) {
      std::string path = JoinPath(hdfs_home, "lib", "native", kLibHdfsDso);
      status_ = TryLoadAndBind(path.c_str(), &handle_);
      if (status_.ok()) {
        return;
      }
    }

    status_ = TryLoadAndBind(kLibHdfsDso, &handle_);
  }

  Status status_;
  void* handle_ = nullptr;
};

class HdfsFileIO : public FileIO {
 public:
  HdfsFileIO();

  ~HdfsFileIO();

  bool Initialize(const ConfigMap& conf) override;

  bool IsDirectory() override;
  std::vector<std::string> ListDirectory(FilterFunc filter) override;

  bool ReadData(void* data, size_t size) override;
  bool WriteData(const void* data, size_t size) override;

  std::unique_ptr<FileIO> Open(Slice path, bool readonly) override;

 private:
  void CheckFileAttr();

 private:
  bool read_;
  bool is_dir_;
  void* buffer_;
  size_t capacity_;
  size_t offset_;
  LibHDFS* hdfs_;
  hdfsFS fs_;
  hdfsFile file_;
  std::string scheme_;
  std::string namenode_;
  std::string path_;
};


HdfsFileIO::HdfsFileIO()
    : FileIO(),
      read_(true),
      is_dir_(false),
      buffer_(nullptr),
      capacity_(0),
      offset_(0),
      hdfs_(LibHDFS::Load()),
      fs_(0),
      file_(0),
      scheme_("hdfs") { }

bool HdfsFileIO::Initialize(const ConfigMap& conf) {
  // Reset state
  initialized_ = false;
  read_ = true;
  if (buffer_ != nullptr) {
    free(buffer_);
  }
  buffer_ = nullptr;
  capacity_ = 0;
  offset_ = 0;
  read_index_ = 0;

  auto it = conf.find("scheme");
  if (it != conf.end()) {
    scheme_ = it->second;
  }

  it = conf.find("namenode");
  if (it != conf.end()) {
    namenode_ = it->second;
  }

  it = conf.find("path");
  if (it == conf.end()) {
    EULER_LOG(ERROR) << "Please specify hdfs dir path";
    return false;
  }
  path_ = it->second;

  it = conf.find("read");
  if (it != conf.end() && (it->second == "false" ||
                           it->second == "no" ||
                           it->second == "0")) {
    read_ = false;
  }

  int flags = O_RDONLY;
  if (!read_) {
    flags = O_WRONLY | O_CREAT;
  }

  auto name_port = Split(namenode_, ":");
  hdfsBuilder* builder = hdfs_->hdfsNewBuilder();
  if (scheme_ == "file") {
    hdfs_->hdfsBuilderSetNameNode(builder, nullptr);
  } else if (scheme_ == "viewfs") {
    char* default_fs = nullptr;
    hdfs_->hdfsConfGetStr("fs.defaultFS", &default_fs);
    Slice default_scheme, default_cluster, default_path;
    ParseURI(default_fs, &default_scheme, &default_cluster, &default_path);

    if (scheme_ != default_scheme ||
        (namenode_ != "" && namenode_ != default_fs)) {
      EULER_LOG(ERROR) <<
          "viewfs is only supported as a fs.defaultFS.";
      return false;
    }

    hdfs_->hdfsBuilderSetNameNode(builder, "default");
  } else {
    hdfs_->hdfsBuilderSetNameNode(
        builder, namenode_ == "" ? "default" : name_port[0].c_str());
    if (name_port.size() == 2) {
      hdfs_->hdfsBuilderSetNameNodePort(builder, atoi(name_port[1].c_str()));
    }
  }

  fs_ = hdfs_->hdfsBuilderConnect(builder);
  if (!fs_) {
    EULER_LOG(ERROR) << "Connect to hdfs host " << namenode_ << " failed!";
    return false;
  }

  CheckFileAttr();

  if (!is_dir_) {
    file_ = hdfs_->hdfsOpenFile(fs_, path_.c_str(), flags, 0, 0, 0);
    if (!file_) {
      EULER_LOG(ERROR) << "Open hdfs file: hdfs://" << namenode_
                       << "/" << path_ << " failed!";
      return false;
    }
  }

  buffer_ = malloc(kBufferSize);
  initialized_ = true;
  EULER_LOG(INFO) << "Open hdfs file: hdfs://" << namenode_
                  << "/" << path_ << " sucessfully!";
  return true;
}

HdfsFileIO::~HdfsFileIO() {
  if (initialized_ && !is_dir_) {
    hdfs_->hdfsCloseFile(fs_, file_);
    if (buffer_ != nullptr) {
      free(buffer_);
    }
  }
}

bool HdfsFileIO::IsDirectory() {
  return is_dir_;
}

std::vector<std::string> HdfsFileIO::ListDirectory(FilterFunc filter) {
  std::vector<std::string> files;
  if (is_dir_) {
    int num_entries = 0;
    auto infos = hdfs_->hdfsListDirectory(fs_, path_.c_str(), &num_entries);
    for (int i = 0; i < num_entries; ++i) {
      std::vector<std::string> parts = Split(infos[i].mName, '/');
      std::string filename = parts[parts.size() - 1];
      if (!filter || filter(filename)) {
        files.emplace_back(filename);
      }
    }
    hdfs_->hdfsFreeFileInfo(infos, num_entries);
  }
  return files;
}

bool HdfsFileIO::ReadData(void* data, size_t size) {
  if (!initialized_ || is_dir_) {
    return false;
  }

  char* dst = static_cast<char*>(data);
  char* src = static_cast<char*>(buffer_) + offset_;
  while (offset_ + size > capacity_) {
    memcpy(dst, src, capacity_ - offset_);

    read_index_ += capacity_ - offset_;
    size -= capacity_ - offset_;
    dst += capacity_ - offset_;
    offset_ = 0;

    capacity_ = hdfs_->hdfsRead(fs_, file_, buffer_, kBufferSize);
    src = static_cast<char*>(buffer_) + offset_;
    if (capacity_ < static_cast<size_t>(kBufferSize)) {
      break;
    }
  }

  if (offset_ + size > capacity_) {
    return false;
  }

  memcpy(dst, src, size);
  offset_ += size;
  read_index_ += size;

  return true;
}

bool HdfsFileIO::WriteData(const void* data, size_t size) {
  if (!initialized_ || is_dir_) {
    return false;
  }

  auto ret = hdfs_->hdfsWrite(fs_, file_, data, size);
  return (static_cast<size_t>(ret) == size);
}

void HdfsFileIO::CheckFileAttr() {
  if (fs_) {
    auto info = hdfs_->hdfsGetPathInfo(fs_, path_.c_str());
    if (info != nullptr) {
      file_size_ = info->mSize;
    }
    is_dir_ = (info != nullptr && info->mKind == kObjectKindDirectory);
    hdfs_->hdfsFreeFileInfo(info, 1);
  }
}

std::unique_ptr<FileIO> HdfsFileIO::Open(Slice path, bool readonly) {
  if (!is_dir_) {
    return nullptr;
  }

  std::unique_ptr<HdfsFileIO> file_io(new HdfsFileIO);
  file_io->read_ = readonly;
  file_io->buffer_ = malloc(kBufferSize);
  file_io->fs_ = fs_;
  file_io->scheme_ = scheme_;
  file_io->namenode_ = namenode_;
  file_io->path_ = JoinPath(path_, path);

  file_io->CheckFileAttr();

  if (!file_io->is_dir_) {
    int flags = O_RDONLY;
    if (!readonly) {
      flags = O_WRONLY | O_CREAT;
    }

    file_io->file_ = hdfs_->hdfsOpenFile(
        file_io->fs_, file_io->path_.c_str(), flags, 0, 0, 0);
    if (!file_io->file_) {
      EULER_LOG(ERROR) << "Open hdfs file: hdfs://" << namenode_
                       << "/" << file_io->path_ << " failed!";
      return nullptr;
    }
  }

  file_io->initialized_ = true;
  return std::move(file_io);
}

REGISTER_FILE_IO("hdfs", HdfsFileIO);
REGISTER_FILE_IO("viewfs", HdfsFileIO);

}  // namespace euler
