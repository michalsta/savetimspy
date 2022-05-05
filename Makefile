PROJECT_NAME = savetimspy
VE = ve_$(PROJECT_NAME)
install: clean_ve
	virtualenv $(VE)
	$(VE)/bin/pip install IPython pytest
	$(VE)/bin/pip install -e .
clean_ve:
	rm -rf $(VE) || true
py:
	$(VE)/bin/ipython
install_without_soft_links:
	rm -rf $(VE)2 || true
	virtualenv $(VE)2
	$(VE)2/bin/pip install IPython pytest
	$(VE)2/bin/pip install .