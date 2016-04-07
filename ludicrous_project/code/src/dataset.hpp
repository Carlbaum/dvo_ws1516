#include <Eigen/Dense>
#include "eigen_typedef.h"
#include "lieAlgebra.hpp"
#include "tum_benchmark.hpp"

//using namespace std;
using namespace Eigen;

struct Frame {
  double timestamp;
  double colorTimestamp;
  double depthTimestamp;
  double groundtruthTimestamp;
  std::string colorPath;
  std::string depthPath;
  Vector6f groundtruthXi;
};

struct GroundtruthRow { double timestamp; Vector6f xi; };

std::vector<GroundtruthRow> loadGroundtruthFile(std::string file) {
  std::vector<GroundtruthRow> rows;
  std::ifstream stream;
  stream.open(file.c_str());
  if (!stream.is_open()) { throw std::runtime_error("Could not open file"); }

  std::string line;
  while (getline(stream, line)) {
    if (line.empty() || line.compare(0, 1, "#") == 0) { continue; }

    GroundtruthRow row;
    float a[7];

    std::istringstream(line)
      >> row.timestamp
      >> a[0] >> a[1] >> a[2] >> a[3] >> a[4] >> a[5] >> a[6];

    Matrix3f rot = Quaternionf(a[6], a[3], a[4], a[5]).toRotationMatrix();
    Vector3f trans(a[0], a[1], a[2]);
    convertTToSE3(row.xi, rot, trans);

    rows.push_back(row);
  }

  stream.close();
  return rows;
}

struct ImageListRow { double timestamp; std::string path; };

std::vector<ImageListRow> loadImageList(std::string file) {
  std::vector<ImageListRow> rows;
  std::ifstream stream;
  stream.open(file.c_str());
  if (!stream.is_open()) { throw std::runtime_error("Could not open file"); }

  std::string line;
  while (getline(stream, line)) {
    if (line.empty() || line.compare(0, 1, "#") == 0) { continue; }

    ImageListRow row;
    std::istringstream(line) >> row.timestamp >> row.path;

    rows.push_back(row);
  }

  stream.close();
  return rows;
}

Eigen::Matrix3f loadK(std::string file) {
  std::vector<float> Ks;
  std::ifstream stream;
  stream.open(file.c_str());
  if (!stream.is_open()) { throw std::runtime_error("Could not open file K.txt"); }

  std::string line;
  while (getline(stream, line)) {
    if (line.empty() || line.compare(0, 1, "#") == 0) { continue; }

    float k;
    std::istringstream(line) >> k;

    Ks.push_back(k);
  }

  stream.close();

  Eigen::Matrix3f out;

  for (int i=0;i<3;i++) {
    out.row(i) << Ks.at(3*i), Ks.at(3*i+1), Ks.at(3*i+2);
  }

  return out;
}

class Dataset {
public:
  Eigen::Matrix3f K;

  Dataset(std::string path) {
    // Load the data from all three files
    std::vector<ImageListRow> rgbRows = loadImageList(path + "/rgb.txt");
    std::vector<ImageListRow> depthRows = loadImageList(path + "/depth.txt");
    std::vector<GroundtruthRow> groundtruthRows = loadGroundtruthFile(path + "/groundtruth.txt");
    K = loadK(path + "/K.txt");

    size_t k = 0, l = 0, m = 0; double tMaxLastPushed = -INFINITY;
    while (true) {
      // Break if it reached the end of one
      if (k >= rgbRows.size()) { break; }
      if (l >= depthRows.size()) { break; }
      if (m >= groundtruthRows.size()) { break; }

      double tRGB = rgbRows[k].timestamp;
      double tDepth = depthRows[l].timestamp;
      double tGround = groundtruthRows[m].timestamp;

      double tMin = min(tGround, min(tRGB, tDepth));
      double tMax = max(tGround, max(tRGB, tDepth));

      // Create a frame if all three entries are new
      if (tMax == tMaxLastPushed) { frames.pop_back(); }
      if (tMaxLastPushed < tMin || tMax == tMaxLastPushed) {
        tMaxLastPushed = tMax;
        frames.push_back((Frame) {
          tMin,
          tRGB,
          tDepth,
          tGround,
          path + "/" + rgbRows[k].path,
          path + "/" + depthRows[l].path,
          groundtruthRows[m].xi
        });
      }

      // Advance the one with the smallest timestamp
      if (tRGB == tMin) { k++; }
      if (tDepth == tMin) { l++; }
      if (tGround == tMin) { m++; }
    }
  }
  std::vector<Frame>frames;
};
