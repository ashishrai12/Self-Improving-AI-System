.PHONY: all python rust julia clean

all: python rust julia

python:
	python pipeline/feedback_loop.py

rust:
	cd src_rust && cargo build --release

julia:
	julia simulations_julia/stability_analysis.jl

clean:
	rm -rf src_rust/target
	find . -type d -name "__pycache__" -exec rm -rf {} +
