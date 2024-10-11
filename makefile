package: .dev-dependencies
	rm -r ./dist || true
	python -m build

upload-pypi: .dev-dependencies
	twine upload --skip-existing dist/* --verbose

upload-testpypi: .dev-dependencies
	twine upload --skip-existing  --repository testpypi dist/* --verbose

.dev-dependencies:
	pip install build twine