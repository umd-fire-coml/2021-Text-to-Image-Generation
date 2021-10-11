import os
from os.path import exists

import pytest

@pytest.mark.parametrize("needed_files", [
    "ind.mat",
    "language_original.mat",
    "subset_index.tar.gz"
])

def test_data(needed_files):
  all_files_available = True
  for path in needed_files:
    if not exists(os.path.join("data", path)):
      all_files_available = False
      break
  assert(all_files_available)
