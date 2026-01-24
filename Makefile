UV_CACHE_DIR ?= $(CURDIR)/uv-cache
export UV_CACHE_DIR

.PHONY: install-data-gen
install-data-gen: ## Install the virtual environment and install the pre-commit hooks.
	@echo "🚀 Creating virtual environment using uv"
	@uv sync --locked --group data-gen
	@uv pip install setuptools flash-attn==2.7.3 --no-build-isolation
	@uv run pre-commit install

.PHONY: install-sft
install-sft: ## Install the virtual environment and install the pre-commit hooks.
	@echo "🚀 Creating virtual environment using uv"
	@uv sync --locked --group sft
	@uv pip install --upgrade setuptools ninja
	@uv pip install --upgrade --torch-backend cu128 "xformers==0.0.33.post1"
	@TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=3 uv pip install --upgrade --no-build-isolation "flash-attn==2.8.3"
	@uv run pre-commit install

.PHONY: install-grpo
install-grpo: ## Install the virtual environment and install the pre-commit hooks.
	@echo "🚀 Creating virtual environment using uv"
	@uv sync --locked --group grpo
	@uv run pre-commit install

.PHONY: setup-dev
setup-dev: ## Setup the development environment
	@echo "🚀 Setting up development environment"
	@uv run linalg_zero/distillation/scripts/push_debug_dataset.py --dataset-name rfvasile/linalg-debug --private

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "🚀 Linting code: Running pre-commit"
ifeq ($(CI),true)
	@echo "🔍 CI detected: Running ruff in check mode"
	@uv run ruff check .
	@uv run ruff format --check .
	@SKIP=ruff,ruff-format uv run pre-commit run -a
else
	@uv run pre-commit run -a
endif
	@echo "🚀 Static type checking: Running mypy"
	@uv run mypy
	@echo "🚀 Checking for obsolete dependencies: Running deptry"
	@uv run deptry .

.PHONY: test
test: ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	@uv run python -m pytest --cov --cov-config=pyproject.toml --cov-report=xml

.PHONY: coverage-site
coverage-site: ## Generate coverage report in HTML format
	@echo "🚀 Generating coverage report in HTML format"
	@uv run coverage html

.PHONY: build
build: clean-build ## Build wheel file
	@echo "🚀 Creating wheel file"
	@uvx --from build pyproject-build --installer uv

.PHONY: clean-build
clean-build: ## Clean build artifacts
	@echo "🚀 Removing build artifacts"
	@uv run python -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None"

.PHONY: publish
publish: ## Publish a release to PyPI.
	@echo "🚀 Publishing."
	@uvx twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

.PHONY: build-and-publish
build-and-publish: build publish ## Build and publish.

.PHONY: docs-test
docs-test: ## Test if documentation can be built without warnings or errors
	@echo "🚀 Testing documentation build"
	@uv run mkdocs build -s

.PHONY: docs
docs: ## Build and serve the documentation
	@echo "🚀 Building and serving documentation"
	@uv run mkdocs serve

.PHONY: semantic-release
semantic-release: ## Test semantic release
	@echo "🚀 Testing semantic release"
	@uv run semantic-release -vv --noop version --print

.PHONY: gh-deploy
gh-deploy: ## Deploy the documentation to GitHub Pages
	@echo "🚀 Deploying documentation to GitHub Pages"
	@uv run mkdocs gh-deploy --force

LLAMACPP_CONFIG=linalg_zero/config/distillation/llamacpp_qwen3_32b_instruct.yaml
VLLM_CONFIG=linalg_zero/config/distillation/vllm_qwen3_32b.yaml

.PHONY: distillation-llamacpp
distillation-llamacpp: ## Start the llama.cpp server
	@echo "🚀 Starting llama.cpp server"
	@INFERENCE_BACKEND=llamacpp uv run python linalg_zero/distillation/launch_server.py --config $(LLAMACPP_CONFIG)

.PHONY: distillation-vllm
distillation-vllm: ## Start the vLLM server
	@echo "🚀 Starting vLLM server"
	@. ./env.sh && uv run python linalg_zero/distillation/launch_server.py --config ${VLLM_CONFIG}

.PHONY: distillation
distillation: ## Run the distillation pipeline using the vllm config
	@echo "🚀 Running distillation pipeline"
	@. ./env.sh && uv run python linalg_zero/distillation.py --config ${VLLM_CONFIG}

.PHONY: distillation-vllm-local
distillation-vllm-local: ## Start the vLLM server
	@echo "🚀 Starting vLLM server"
	@export USING_VLLM=true INFERENCE_BACKEND=vllm && uv run python linalg_zero/distillation/launch_server.py --config linalg_zero/config/distillation/vllm_qwen3_4b_think.yaml


.PHONY: distillation-local
distillation-debug: ## Start the vLLM server
	@echo "🚀 Starting vLLM server"
	@export USING_VLLM=true && uv run python linalg_zero/distillation.py --config linalg_zero/config/distillation/vllm_qwen3_4b_think.yaml


# SFT Training Commands
SFT_CONFIG=linalg_zero/config/sft/sft_debug_config.yaml
# SFT_CONFIG=linalg_zero/config/sft/sft_config.yaml
ACCELERATE_CONFIG=linalg_zero/config/sft/accelerate/zero3.yaml

.PHONY: sft-debug
sft-debug: ## Run SFT training on single GPU
	@echo "🚀 Running SFT training on single GPU"
	@uv run python linalg_zero/sft.py --config $(SFT_CONFIG)


.PHONY: sft-distributed
sft-distributed: ## Run SFT training with distributed setup using DeepSpeed ZeroStage 3
	@echo "🚀 Running distributed SFT training with DeepSpeed"
	@uv run accelerate launch --config_file=$(ACCELERATE_CONFIG) linalg_zero/sft.py --config $(SFT_CONFIG)

.PHONY: prepare-grpo-dataset
prepare-grpo-dataset: ## Prepare the GRPO dataset
	@echo "🚀 Creating GRPO dataset"
	@uv run linalg_zero/grpo/process_dataset.py

.PHONY: generate-optimised-config
generate-optimised-config: ## Generate the optimised config
	@echo "🚀 Generating optimised config"
	@uv run linalg_zero/generator/analysis/analyse.py

.PHONY: run-training
run-training: ## Run the training pipeline
	@echo "🚀 Running training pipeline"
	@$(MAKE) setup-dev
	@$(MAKE) prepare-grpo-dataset
	@echo "🚀 Training pipeline completed"


.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
