#ifndef RECORD_H_
#define RECORD_H_

struct Record {
 public:
  Record();
  Record(const int uid, const int iid, const float r);
  ~Record();
  int user_id;
  int item_id;
  float rating;
};

#endif

