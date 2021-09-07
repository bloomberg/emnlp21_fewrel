#!/bin/bash -e

# check out a snapshot from the luke repository.
git submodule init
git submodule update

# Apply changes to make luke compatible with huggingface transformers version used in this repo.
git -C luke apply ../luke.diff
cp -r luke/luke/ fewrel/fewshot_re_kit
