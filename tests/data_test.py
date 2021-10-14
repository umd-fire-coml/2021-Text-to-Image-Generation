import os
from os.path import exists

import pytest

@pytest.mark.parametrize("needed_file", [
    "ind.mat",
    "language_original.mat",
    "subset_index.tar.gz"
])

def test_data(needed_file):
  print (os.path.join("data", needed_file))
  assert(exists(os.path.join("data", needed_file)))
