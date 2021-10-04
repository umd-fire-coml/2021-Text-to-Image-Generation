from os.path import exists

def test_data():
  needed_files = ["/ind.mat", "language_original.mat", "subset_index.tar.gz"]
  all_files_available = True
  for path in needed_files:
    if not exists("/data" + path):
      all_files_available = False
      break
    assert(all_files_available)
