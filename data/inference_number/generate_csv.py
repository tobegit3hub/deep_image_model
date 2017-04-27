#!/usr/bin/env python

import os
import random
import time


def rule_add_one(start_number):
  return start_number + 1


def rule_multiple_two(start_number):
  return start_number * 2


def rule(mode, start_number):
  if mode == "add_one":
    return rule_add_one(start_number)
  elif mode == "multiple_two":
    return rule_multiple_two(start_number)
  return None


def main():
  # Define parameters
  #mode = "add_one"
  mode = "multiple_two"
  FEATURE_SIZE = 9
  MAX_START_NUMBER = 1000
  TRAIN_SET_SIZE = 90000
  #TRAIN_SET_SIZE = 10
  VALIDATE_SET_SIZE = 10000
  TRAIN_SET_FILENAME = os.path.join("./" + mode + "/train.csv")
  VALIDATE_SET_FILENAME = os.path.join("./" + mode + "/validate.csv")
  if not os.path.exists("./" + mode):
  	os.makedirs("./" + mode)

  train_set_start_numbers = [random.randint(0, MAX_START_NUMBER)
                             for i in range(TRAIN_SET_SIZE)]
  validate_set_start_numbers = [random.randint(0, MAX_START_NUMBER)
                                for i in range(VALIDATE_SET_SIZE)]

  train_set_content = ""
  validate_set_content = ""
  start_time = time.time()

  # Generate train set content
  for i in train_set_start_numbers:
    line_string = str(i) + ","
    next_number = i

    for j in range(FEATURE_SIZE):
      next_number = rule(mode, next_number)
      line_string += str(next_number) + ","

    train_set_content += line_string[:-1] + "\n"

  # Generate validate set content
  for i in validate_set_start_numbers:
    line_string = str(i) + ","
    next_number = i

    for j in range(FEATURE_SIZE):
      next_number = rule(mode, next_number)
      line_string += str(next_number) + ","

    validate_set_content += line_string[:-1] + "\n"

  # Write csv content to files
  with open(TRAIN_SET_FILENAME, "w") as f:
    f.write(train_set_content)
  with open(VALIDATE_SET_FILENAME, "w") as f:
    f.write(validate_set_content)

  print("[{}]Generate CSV files: {} and {}".format(time.time(
  ) - start_time, TRAIN_SET_FILENAME, VALIDATE_SET_FILENAME))


if __name__ == "__main__":
  main()
