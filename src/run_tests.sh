#!/usr/bin/env bash
# run_tests.sh — Run pipeline unit tests with collapsible per-file output
#
# Usage:
#   ./run_tests.sh            — compact view: counts + failed test names only
#   ./run_tests.sh -v         — verbose: show every test name per file
#   ./run_tests.sh -f         — failures only: skip files with 100% pass rate
#   ./run_tests.sh --include ingestion,aggregation
#   ./run_tests.sh --skip ingestion
#   ./run_tests.sh --skip test_ingestion.py,test_pipeline

set -euo pipefail

VENV_PYTHON="../.venv/bin/python"
TEST_DIR="tests"
VERBOSE=0
FAILURES_ONLY=0
DEFAULT_SKIP_MODULES=("pipeline")
SKIP_MODULES=()
INCLUDE_MODULES=()

print_help() {
    cat <<'EOF'
run_tests.sh — Run pipeline unit tests with collapsible per-file output

Usage:
  ./run_tests.sh                   compact view: counts + failed test names only
  ./run_tests.sh -v                verbose: show every test name per file
  ./run_tests.sh -f                failures only: skip files with 100% pass rate
  ./run_tests.sh --include ingestion,aggregation
  ./run_tests.sh --skip ingestion  skip module(s) by component or filename

Flags:
  -v, --verbose                    show full per-test output
  -f, --failures                   only expand failing modules
  -i, --include LIST               include only module(s), repeatable or comma-separated
                                   examples: ingestion, test_ingestion.py
                                   include overrides default skips
  -s, --skip LIST                  skip module(s), repeatable or comma-separated
                                   examples: ingestion, test_ingestion.py
                                   default: pipeline
  -h, --help                       show this help
EOF
}

add_skip_values() {
    local raw="$1"
    local token
    local old_ifs
    local trimmed

    old_ifs="$IFS"
    IFS=','
    for token in $raw; do
        # Trim leading/trailing whitespace (bash 3.2 compatible)
        trimmed=$(echo "$token" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
        [[ -n "$trimmed" ]] && SKIP_MODULES+=("$trimmed")
    done
    IFS="$old_ifs"
}

add_include_values() {
    local raw="$1"
    local token
    local old_ifs
    local trimmed

    old_ifs="$IFS"
    IFS=','
    for token in $raw; do
        # Trim leading/trailing whitespace (bash 3.2 compatible)
        trimmed=$(echo "$token" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
        [[ -n "$trimmed" ]] && INCLUDE_MODULES+=("$trimmed")
    done
    IFS="$old_ifs"
}

test_matches_selector() {
    local test_file="$1"
    local component="$2"
    local selector="$3"
    local test_base
    local normalized

    test_base=$(basename "$test_file")
    normalized="$selector"
    normalized="${normalized##*/}"     # basename
    normalized="${normalized%.py}"     # strip .py
    normalized="${normalized#test_}"   # strip test_ prefix

    if [[ "$test_file" == "$selector" || "$test_base" == "$selector" || "$component" == "$selector" || "$component" == "$normalized" ]]; then
        return 0
    fi

    return 1
}

should_include_test() {
    local test_file="$1"
    local component="$2"
    local include

    if [[ ${#INCLUDE_MODULES[@]} -eq 0 ]]; then
        return 0
    fi

    for include in "${INCLUDE_MODULES[@]}"; do
        if test_matches_selector "$test_file" "$component" "$include"; then
            return 0
        fi
    done

    return 1
}

should_skip_test() {
    local test_file="$1"
    local component="$2"
    local skip
    local default_skip

    # Explicit --skip entries always win.
    if [[ ${#SKIP_MODULES[@]} -gt 0 ]]; then
        for skip in "${SKIP_MODULES[@]}"; do
            if test_matches_selector "$test_file" "$component" "$skip"; then
                return 0
            fi
        done
    fi

    # If caller explicitly included this module, do not apply default skips.
    if [[ ${#INCLUDE_MODULES[@]} -gt 0 ]] && should_include_test "$test_file" "$component"; then
        return 1
    fi

    if [[ ${#DEFAULT_SKIP_MODULES[@]} -eq 0 ]]; then
        return 1
    fi

    for default_skip in "${DEFAULT_SKIP_MODULES[@]}"; do
        if test_matches_selector "$test_file" "$component" "$default_skip"; then
            return 0
        fi
    done

    return 1
}

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -f|--failures)
            FAILURES_ONLY=1
            shift
            ;;
        -i|--include)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for $1"
                echo ""
                print_help
                exit 1
            fi
            add_include_values "$2"
            shift 2
            ;;
        --include=*)
            add_include_values "${1#*=}"
            shift
            ;;
        -s|--skip)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for $1"
                echo ""
                print_help
                exit 1
            fi
            add_skip_values "$2"
            shift 2
            ;;
        --skip=*)
            add_skip_values "${1#*=}"
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            print_help
            exit 1
            ;;
    esac
done

# ── Colors ───────────────────────────────────────────────────────────────────
BOLD="\033[1m"
RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
DIM="\033[2m"
RESET="\033[0m"

# ── Collect test files (bash 3.2 compatible) ─────────────────────────────────
TEST_FILES=()
while IFS= read -r f; do
    TEST_FILES+=("$f")
done < <(find "$TEST_DIR" -name "test_*.py" | sort)

if [[ ${#TEST_FILES[@]} -eq 0 ]]; then
    echo "No test files found in $TEST_DIR"
    exit 1
fi

total_passed=0
total_failed=0
declare -a summary_lines
declare -a skipped_lines

# ── Per-file loop ─────────────────────────────────────────────────────────────
for test_file in "${TEST_FILES[@]}"; do
    component=$(basename "$test_file" .py | sed 's/^test_//')

    if ! should_include_test "$test_file" "$component"; then
        continue
    fi

    if should_skip_test "$test_file" "$component"; then
        skipped_lines+=("${YELLOW}  ⏭  $(printf '%-28s' "$component")  skipped${RESET}")
        continue
    fi

    # Run pytest — short traceback so failure output stays readable
    raw_output=$("$VENV_PYTHON" -m pytest "$test_file" -v --tb=line --no-header 2>&1 || true)

    passed=$(echo "$raw_output" | grep -c " PASSED" || true)
    failed=$(echo "$raw_output" | grep -c " FAILED" || true)

    total_passed=$((total_passed + passed))
    total_failed=$((total_failed + failed))

    # Skip all-green files when -f flag is set
    [[ $FAILURES_ONLY -eq 1 && $failed -eq 0 ]] && {
        summary_lines+=("${GREEN}  ✓  $(printf '%-28s' "$component")  passed: $(printf '%3d' "$passed")  failed:   0${RESET}")
        continue
    }

    # ── Section header ────────────────────────────────────────────────────────
    if [[ $failed -gt 0 ]]; then
        hdr_color="$RED";   icon="✗"
    else
        hdr_color="$GREEN";  icon="✓"
    fi

    echo ""
    echo -e "${hdr_color}${BOLD}  ${icon}  ${component}${RESET}  ${DIM}passed: ${passed}  failed: ${failed}${RESET}"
    echo -e "${DIM}  $(printf '─%.0s' {1..58})${RESET}"

    if [[ $VERBOSE -eq 1 ]]; then
        # ── Verbose: every test line ──────────────────────────────────────────
        while IFS= read -r line; do
            if   [[ "$line" == *" PASSED"* ]]; then
                echo -e "    ${GREEN}${line}${RESET}"
            elif [[ "$line" == *" FAILED"* ]]; then
                echo -e "    ${RED}${line}${RESET}"
            elif [[ "$line" == "FAILED "* || "$line" == "  "* ]]; then
                echo -e "    ${DIM}${line}${RESET}"
            fi
        done <<< "$raw_output"

    else
        # ── Compact: one line per failure from pytest's summary block ─────────
        # The summary lines look like:  FAILED path::Class::method - ErrorType
        if [[ $failed -gt 0 ]]; then
            while IFS= read -r line; do
                if [[ "$line" == "FAILED "* ]]; then
                    # Strip leading "FAILED " and split on " - "
                    rest="${line#FAILED }"
                    test_path="${rest%% - *}"
                    # Shorten to just Class::method
                    short=$(echo "$test_path" | sed 's|.*::||')
                    class=$(echo "$test_path" | sed 's|.*::\(.*\)::\(.*\)|\1|')
                    reason=""
                    if [[ "$rest" == *" - "* ]]; then
                        reason="  ${DIM}→ ${rest##* - }${RESET}"
                    fi
                    echo -e "    ${RED}✗${RESET}  ${DIM}${class}::${short}${reason}${RESET}"
                fi
            done <<< "$raw_output"
        else
            echo -e "    ${DIM}All ${passed} tests passed${RESET}"
        fi
    fi

    # Accumulate summary row
    if [[ $failed -gt 0 ]]; then
        summary_lines+=("${RED}  ✗  $(printf '%-28s' "$component")  passed: $(printf '%3d' "$passed")  failed: $(printf '%3d' "$failed")${RESET}")
    else
        summary_lines+=("${GREEN}  ✓  $(printf '%-28s' "$component")  passed: $(printf '%3d' "$passed")  failed:   0${RESET}")
    fi
done

# ── Overall summary ───────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}  $(printf '═%.0s' {1..58})${RESET}"
echo -e "${BOLD}    RESULTS BY COMPONENT${RESET}"
echo -e "${BOLD}  $(printf '─%.0s' {1..58})${RESET}"
if [[ ${#summary_lines[@]} -gt 0 ]]; then
    for line in "${summary_lines[@]}"; do
        echo -e "$line"
    done
fi
if [[ ${#skipped_lines[@]} -gt 0 ]]; then
    for line in "${skipped_lines[@]}"; do
        echo -e "$line"
    done
fi
echo -e "${BOLD}  $(printf '─%.0s' {1..58})${RESET}"

if [[ $total_failed -gt 0 ]]; then
    totals_color="$RED"
else
    totals_color="$GREEN"
fi
echo -e "${totals_color}${BOLD}    TOTAL  passed: ${total_passed}  failed: ${total_failed}${RESET}"
echo -e "${BOLD}  $(printf '═%.0s' {1..58})${RESET}"
echo ""

if [[ $VERBOSE -eq 0 && $FAILURES_ONLY -eq 0 ]]; then
    echo -e "${DIM}  Tip: run with -v for all test names, -f to show only failing files${RESET}"
    echo ""
fi

# Exit non-zero if any tests failed
[[ $total_failed -eq 0 ]]
