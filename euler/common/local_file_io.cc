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

#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>

#include <fstream>
#include <string>
#include <vector>

#include "euler/common/logging.h"
#include "euler/common/file_io.h"

namespace euler {

class LocalFileIO : public FileIO {
 public:
  LocalFileIO();

  virtual ~LocalFileIO();

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
  std::string path_;
  std::fstream file_;
};

LocalFileIO::LocalFileIO(): FileIO(), read_(true), is_dir_(false) {
}

LocalFileIO::~LocalFileIO() {
  file_.close();
}

bool LocalFileIO::Initialize(const ConfigMap& conf) {
  initialized_ = false;
  read_ = true;

  auto it = conf.find("path");
  if (it == conf.end()) {
    EULER_LOG(ERROR) << "Please specify filename in config";
    return false;
  }
  path_ = it->second;

  it = conf.find("read");
  if (it != conf.end() && (it->second == "false" ||
                           it->second == "no" ||
                           it->second == "0")) {
    read_ = false;
  }

  CheckFileAttr();

  if (!is_dir_) {
    std::ios::openmode mode = std::fstream::in | std::fstream::binary;
    if (!read_) {
      mode = std::fstream::out | std::fstream::binary;
    }
    file_.open(path_.c_str(), mode);
    if (!file_.is_open()) {
      EULER_LOG(ERROR) << "file :" << path_ << " open failed";
      return false;
    }
  }

  initialized_ = true;
  return true;
}

bool LocalFileIO::IsDirectory() {
  return is_dir_;
}

std::vector<std::string> LocalFileIO::ListDirectory(FilterFunc filter) {
  std::vector<std::string> files;
  if (is_dir_) {
    DIR* dir = opendir(path_.c_str());
    if (dir != nullptr) {
      struct dirent* ent = NULL;
      while ((ent = readdir(dir)) != NULL)  {
        std::string filename(ent->d_name);
        if (filename != "." && filename != "..") {
          if (!filter || filter(filename)) {
            files.push_back(filename);
          }
        }
      }
      closedir(dir);
    }
  }

  return files;
}

bool LocalFileIO::ReadData(void* data, size_t size) {
  if (!initialized_ || is_dir_) {
    return false;
  }
  file_.read(static_cast<char*>(data), size);
  read_index_ += size;
  return file_.good();
}

bool LocalFileIO::WriteData(const void* data, size_t size) {
  if (!initialized_ || is_dir_) {
    return false;
  }
  file_.write(static_cast<const char*>(data), size);
  return file_.good();
}

void LocalFileIO::CheckFileAttr() {
  struct stat st;
  if (stat(path_.c_str(), &st) < 0) {
    return;
  }

  file_size_ = st.st_size;
  is_dir_ = S_ISDIR(st.st_mode);
}

std::unique_ptr<FileIO> LocalFileIO::Open(Slice path, bool readonly) {
  if (!is_dir_) {
    return nullptr;
  }

  ConfigMap config;
  std::unique_ptr<FileIO> file_io(new LocalFileIO);
  std::string file_path = JoinPath(path_, path);
  config["path"] = file_path;
  config["read"] = readonly ? "true" : "false";

  if (!file_io->Initialize(config)) {
    return nullptr;
  }

  return file_io;
}


REGISTER_FILE_IO("[local]", LocalFileIO);
REGISTER_FILE_IO("file", LocalFileIO);

}  // namespace euler
