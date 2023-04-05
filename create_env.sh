#!/bin/bash

set -eu

PROJECT_ROOT_DIR=$(dirname "$0")

conda env create --file "${PROJECT_ROOT_DIR}"/env.yaml --prefix "${PROJECT_ROOT_DIR}"/env 2>&1 | tee .conda_create.log
