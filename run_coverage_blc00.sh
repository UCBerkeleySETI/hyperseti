rm coverage.xml
py.test --verbose --cov=hyperseti test/
codecov -t $CODECOV_TOKEN
