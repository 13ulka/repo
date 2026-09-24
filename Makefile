# SceneForge convenience targets (macOS, Apple Silicon). Run from the repo root.
SCENE ?= rumi_room
CLIPS ?=

.PHONY: doctor setup setup-envs test run status preview

doctor:            ## read-only system + clip check
	scripts/doctor.sh $(CLIPS)

setup:             ## create envs from locks + download models (asks first)
	scripts/setup_mac.sh

setup-envs:        ## envs only, no model downloads
	scripts/setup_mac.sh --no-models

test:              ## unit + synthetic end-to-end tests (no models needed)
	.venv/bin/python -m pytest -q

run:               ## make run CLIPS='"/path/a.mkv" "/path/b.mkv"' SCENE=rumi_room
	.venv/bin/sceneforge run $(CLIPS) --scene $(SCENE)

status:
	.venv/bin/sceneforge status --scene $(SCENE)

preview:
	.venv/bin/sceneforge preview --scene $(SCENE)
