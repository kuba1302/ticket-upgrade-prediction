#!/usr/bin/env bash

black . --check
isort --profile black . --check-only