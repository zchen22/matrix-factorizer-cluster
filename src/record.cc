#include "record.h"

Record::Record() : user_id(-1), item_id(-1), rating(0) {
}

Record::Record(const int uid, const int iid, const float r)
    : user_id(uid), item_id(iid), rating(r) {
}

Record::~Record() {
}

