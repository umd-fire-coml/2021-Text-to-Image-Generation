from os.path import exists

def test_data():
  needed_files = ["/captions/cap.dress.test.json", "/captions/cap.dress.train.json", "/captions/cap.dress.val.json", "/captions/cap.shirt.test.json", "/captions/cap.shirt.train.json", "/captions/cap.shirt.val.json", "/captions/cap.toptee.test.json", "/captions/cap.toptee.train.json", "/captions/cap.toptee.val.json", "/image_splits/split.dress.test.json", "/image_splits/split.dress.train.json", "/image_splits/split.dress.json", "/image_splits/split.cap.shirt.test.json", "/image_splits/split.shirt.train.json", "/image_splits/split.shirt.val.json", "/image_splits/split.toptee.test.json", "/image_splits/split.toptee.train.json", "/image_splits/split.toptee.val.json", "/image_url/asin2url.dress.txt", "/image_url/asin2url.shirt.txt", "/image_url/asin2url.topfseftee.txt"]
  all_files_available = True
  for path in needed_files:
    if not exists("/data" + path):
      all_files_available = False
      break
    assert(all_files_available)
