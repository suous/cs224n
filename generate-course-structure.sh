#!/bin/bash

: "${DEBUG:="false"}"
: "${DRY_RUN:="false"}"
: "${VERBOSE:="false"}"
: "${DOWNLOAD:="false"}"

COURSE_NAME="cs224n"
COURSE_URL="https://web.stanford.edu/class/cs224n/"

help() {
  echo "Generate stanford cs224n course structure."
  echo "Note: This script requires curl and htmlq, and will download course assignments from ${COURSE_URL}."
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  -h, --help    output usage information"
  echo "  -d, --dry-run dry run without generate folders"
  echo "  -v, --verbose verbose output"
  echo "      --download download course assignments"
  echo ""
  echo "Examples:"
  echo "  $0 -v --dry-run"
}

# Set debug if desired
if [ "${DEBUG}" == "true" ]; then
  set -x
fi

set -u
while [[ $# -gt 0 ]]; do
  key="${1}"
  case ${key} in
    -h|--help)
      help
      exit 0
      ;;
    -d|--dry-run)
      DRY_RUN="true"
      shift
      ;;
    -v|--verbose)
      VERBOSE="true"
      shift
      ;;
    --download)
      DOWNLOAD="true"
      shift
      ;;
    *)
      echo "Unknown option: ${key}"
      help
      exit 1
      ;;
  esac
done
set +u

if [ "${VERBOSE}" == "true" ]; then
  echo "=======================================  INFO  ======================================="
  echo "Bash version: ${BASH_VERSION}"
  echo "    Course:   ${COURSE_NAME}"
  echo "    Dry run:  ${DRY_RUN}"
  echo "=======================================  INFO  ======================================="
  echo ""
fi

has() {
  type "${1}" > /dev/null 2>&1
}

verify() {
  if ! has curl; then
    echo "curl is required"
    exit 1
  fi

  if ! has htmlq; then
    echo "htmlq is required"
    exit 1
  fi
}

normalize() {
  # normalize course title.
  # 1. lowercase
  # 2. remove punctuations
  # 3. replace space with dash
  # 4. remove duplicate dashes
  # Example: CS224N: Natural Language Processing with Deep Learning -> cs224n-natural-language-processing-with-deep-learning
  echo "${1}" | xargs echo -n | tr '[:upper:]' '[:lower:]' | tr -d '[:punct:]' | tr ' ' '-' | tr -s '-'
}

generate() {
  local course_url="${1}"
  temp_dir=$(mktemp -d) || exit $?
  # shellcheck disable=SC2064
  trap "command rm -rf '${temp_dir}'" EXIT

  if [ "${VERBOSE}" == "true" ]; then
    echo "Getting courses content from ${course_url} ..."
    echo ""
  fi

  course_content="$(curl -s "${course_url}")"
  course_header="$(echo "${course_content}" | htmlq div#header)"
  course_title="$(echo "${course_header}" | htmlq h1 --text | sed 's/:/ -/')"
  course_season="$(echo "${course_header}" | htmlq h3 --text | tr '/' '-')"
  course_year="$(echo "${course_season}" | grep -oE '\d{4}')"
  normalized_course_title="$(normalize "${course_title}")"
  # if course year is not found, use current year
  if [ -z "${course_year}" ]; then
    course_year="$(date +%Y)"
  fi

  course_schedule="$(echo "${course_content}" | htmlq div#schedule)"
  course_table="$(echo "${course_schedule}" | htmlq table.table | htmlq td)"
  course_code_links="$(echo "${course_table}" | htmlq a --attribute href | grep -E ".*assignments/a\d+.zip")"
  course_handout_links="$(echo "${course_table}" | htmlq a --attribute href | grep -E ".*assignments/a\d+.*latex.*.zip")"

  assignment_folder="${normalized_course_title}/${course_year}/assignments"
  if [ "${DRY_RUN}" == "false" ]; then
    declare -a course_links_array=("${course_code_links}" "${course_handout_links}")
    declare -a course_types_array=("code" "handout")

    for i in "${!course_types_array[@]}"; do
      resource_type="${course_types_array[$i]}"
      resource_links="${course_links_array[$i]}"

      for link in ${resource_links}; do
        link_name="$(echo "${link}" | grep -oE "a\d+")"
        if [[ "${link}" != *"${course_url}"* ]]; then
          link="${course_url}${link}"
        fi
        resource_folder="${assignment_folder}/${link_name}/${resource_type}"
        mkdir -p "${resource_folder}"

        if [ "${DOWNLOAD}" == "true" ]; then
          if [ "${VERBOSE}" == "true" ]; then
            echo "Downloading ${link} to ${temp_dir} ..."
          fi
          temp_file="${temp_dir}/${link_name}.zip"
          curl -s -o "${temp_file}" "${link}"
          if [ "${VERBOSE}" == "true" ]; then
            echo "Unzipping ${temp_file} to ${resource_folder} ..."
          fi
          unzip -q -o -j "${temp_file}" -d "${resource_folder}"
        fi
      done
    done
  fi

  course_readme_name="README.md"
  echo "Generating README.md ..."

  course_readme_content="<h1 align=\"center\"><a href=\"${course_url}\">${course_title}</a></h1>\n\n"
  course_readme_content+="<p align=\"center\"><b>${course_season}</b></p>\n\n"
  course_readme_content+="## Assignments\n"

  course_assignments=$(echo "${course_content}" | htmlq div#coursework | htmlq ul | htmlq li --text | grep -E "^Assignment \d+.*" | sort -u)
  declare -a course_assignment_array
  while IFS= read -r line; do
    course_assignment_array+=("${line}")
  done <<< "${course_assignments}"

  course_index=0
  for d in "${assignment_folder}"/* ; do
    course_assignment=${course_assignment_array[${course_index}]}
    course_readme_content="${course_readme_content}\n- [${course_assignment}](${d})"

    code_dir="${d}/code"
    handout_dir="${d}/handout"
    if [ -d "${code_dir}" ]; then
      course_readme_content="${course_readme_content}\n  - [Code](${d}/code)"
    fi
    if [ -d "${handout_dir}" ]; then
      course_readme_content="${course_readme_content}\n  - [Handout](${d}/handout)"
    fi
    course_index=$((course_index + 1))
  done

  if [ "${VERBOSE}" == "true" ]; then
    echo ""
    echo "=============================== ${course_readme_name} ==============================="
    echo -e "${course_readme_content}"
    echo "=============================== ${course_readme_name} ==============================="
    echo ""
  fi
  if [ "${DRY_RUN}" == "false" ]; then
    echo -e "${course_readme_content}" > "${course_readme_name}"
  fi
}

verify

generate "${COURSE_URL}"
