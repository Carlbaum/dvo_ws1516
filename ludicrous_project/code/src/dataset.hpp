#include <Eigen/Dense>
#include "eigen_typedef.h"
#include "lieAlgebra.hpp"
#include "tum_benchmark.hpp"

using namespace std;
using namespace Eigen;

struct Frame {
  double timestamp;
  double colorTimestamp;
  double depthTimestamp;
  double groundtruthTimestamp;
  string colorPath;
  string depthPath;
  Vector6f groundtruthXi;
};

struct GroundtruthRow { double timestamp; Vector6f xi; };

vector<GroundtruthRow> loadGroundtruthFile(string file) {
  vector<GroundtruthRow> rows;
  ifstream stream;
  stream.open(file.c_str());
  if (!stream.is_open()) { throw runtime_error("Could not open file"); }

  string line;
  while (getline(stream, line)) {
    if (line.empty() || line.compare(0, 1, "#") == 0) { continue; }

    GroundtruthRow row;
    float a[7];

    istringstream(line)
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

struct ImageListRow { double timestamp; string path; };

vector<ImageListRow> loadImageList(string file) {
  vector<ImageListRow> rows;
  ifstream stream;
  stream.open(file.c_str());
  if (!stream.is_open()) { throw runtime_error("Could not open file"); }

  string line;
  while (getline(stream, line)) {
    if (line.empty() || line.compare(0, 1, "#") == 0) { continue; }

    ImageListRow row;
    istringstream(line) >> row.timestamp >> row.path;

    rows.push_back(row);
  }

  stream.close();
  return rows;
}

Eigen::Matrix3f loadK(string file) {
  vector<float> Ks;
  ifstream stream;
  stream.open(file.c_str());
  if (!stream.is_open()) { throw runtime_error("Could not open file"); }

  string line;
  while (getline(stream, line)) {
    if (line.empty() || line.compare(0, 1, "#") == 0) { continue; }

    float k;
    istringstream(line) >> k;

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

  Dataset(string path) {
    // Load the data from all three files
    vector<ImageListRow> rgbRows = loadImageList(path + "/rgb.txt");
    vector<ImageListRow> depthRows = loadImageList(path + "/depth.txt");
    vector<GroundtruthRow> groundtruthRows = loadGroundtruthFile(path + "/groundtruth.txt");
    K = loadK(path + "/K.txt");

    int k = 0, l = 0, m = 0; double tMaxLastPushed = -INFINITY;
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
  vector<Frame>frames;
};
