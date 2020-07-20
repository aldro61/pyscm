package: .dev-dependencies
	rm -r ./dist ./build
	python setup.py sdist
	python setup.py bdist_wheel

upload-pypi: .dev-dependencies
	twine upload --skip-existing dist/* --verbose

upload-testpypi: .dev-dependencies
	twine upload --skip-existing  --repository testpypi dist/* --verbose

.dev-dependencies:
	pip install wheel twine