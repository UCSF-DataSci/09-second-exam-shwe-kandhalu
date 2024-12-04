#!/bin/bash

python3 generate_dirty_data.py

# clean the file
# remove comments, empty lines, extra commas, and extract essential columns: 
# patient_id, visit_date, age, education_level, walking_speed
cat ms_data_dirty.csv | grep -v '^#' | sed '/^$/d' | sed -e 's/,,*/,/g' | cut -d ',' -f1,2,4,5,6 > ms_data.csv

# create list file for insurance with tiers
echo -e "Basic\nPremium\nPlatinum" > insurance.lst # create list

visits=$(wc -l < ms_data.csv)

# summary of processed data
echo "Total number of visits: $((visits - 1))" # count visits w/o header

# display first few records
echo "First few records:" 
head -n 5 ms_data.csv

chmod +x prepare.sh