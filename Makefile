clean: clean-pyc clean-test
quality: set-style-dep check-quality
style: set-style-dep set-style

##### basic #####
set-git:
	git config --local commit.template .gitmessage
	git update-index --skip-worktree ./config/config.yaml

set-style-dep:
	pip3 install click==8.0.4 isort==5.13.2 black==24.8.0 flake8==7.1.1

set-style:
	black --config pyproject.toml .
	isort --settings-path pyproject.toml .
	flake8 .

check-quality:
	black --config pyproject.toml --check .
	isort --settings-path pyproject.toml --check-only .

#####  clean  #####
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -f .coverage
	rm -f .coverage.*
	rm -rf .pytest_cache
	rm -rf .mypy_cache