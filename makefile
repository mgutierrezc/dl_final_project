chmod +x run.sh

make_venv:
	python3 -m venv ./env

install_src:
	pip install -e .

get_slurmlauncher:
	git clone https://github.com/nng555/slurmlauncher
	cd slurmlauncher && pip install -e .