.PHONY: profile profile-speedscope install-profile-deps which-pyspy

# Output files
PROFILE_SVG ?= flame.svg
SPEEDSCOPE_JSON ?= profile.speedscope.json

# Try to find py-spy even if installed with --user
PYSPY ?= py-spy
USER_BIN := $(shell python3 -c 'import site; print(site.USER_BASE + "/bin")' 2>/dev/null)
ifeq ($(wildcard $(USER_BIN)/py-spy),$(USER_BIN)/py-spy)
  PYSPY := $(USER_BIN)/py-spy
endif

# Command to run (edit to suit your paths/files)
PROFILE_CMD ?= goproverlay \
	--video /Users/charlieturner/Downloads/output.MP4 \
	--fit satbike.fit \
	--widget power --widget speed --widget gps \
	--output ./Overlayed.mp4 \
	--gpx /Users/charlieturner/Downloads/GX021750_1757722703417.gpx

# Native stack sampling is not supported by py-spy on all platforms
# Default: enable on Linux, disable elsewhere. Override with NATIVE=1/0.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
  DEFAULT_NATIVE := 1
else
  DEFAULT_NATIVE := 0
endif
NATIVE ?= $(DEFAULT_NATIVE)
ifeq ($(NATIVE),1)
  NATIVE_FLAG := --native
else
  NATIVE_FLAG :=
endif

install-profile-deps:
	python3 -m pip install --user py-spy
	@echo "If 'py-spy' is not found, add this to your shell:"
	@python3 - <<'PY'
	import site
	print(f"  export PATH=\"{site.USER_BASE}/bin:$PATH\"")
	PY

profile:
	$(PYSPY) record $(NATIVE_FLAG) --rate 250 -o $(PROFILE_SVG) -- $(PROFILE_CMD)
	@echo "Flame graph written to $(PROFILE_SVG)"

profile-speedscope:
	$(PYSPY) record --format speedscope --rate 250 -o $(SPEEDSCOPE_JSON) -- $(PROFILE_CMD)
	@echo "Speedscope profile written to $(SPEEDSCOPE_JSON)"

which-pyspy:
	@echo Using PYSPY = $(PYSPY)
