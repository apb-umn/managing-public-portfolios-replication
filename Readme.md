Readme for "Managing Public Portfolios"
=======================================

Contents
--------
- Notebooks/   Python (Jupyter) notebooks for Section 5 and Appendix C
- ABN/         Julia code for Section 6 (Angeletos-Buera-Nicolini model)
- Data/        Input data used by the notebooks
- Manuscript/  LaTeX source and figures of the published version

Software requirements
---------------------
Python: the notebooks were written for Python 3.11.5 and were last verified with Python 3.12.
Install the pinned packages in Notebooks/requirements.txt, for example

    python -m venv venv && source venv/bin/activate
    pip install -r Notebooks/requirements.txt

Two pins matter: tikzplotlib needs matplotlib < 3.8, and xlrd is needed by pandas to read Data/barrotaxdata.xls.
The notebooks download macro series from FRED at runtime, so an internet connection is required.

Julia: the ABN code was last verified with Julia 1.12. ABN/Project.toml and ABN/Manifest.toml
list the packages used (CSV, DataFrames, JuMP, OSQP, Ipopt, NLsolve, Parameters, LaTeXStrings,
QuantEcon, Roots, StatsBase, Plots). From the ABN folder run

    julia --project=. -e 'import Pkg; Pkg.instantiate()'
    julia --project=. main_ABN_Nshocks.jl

Section 5 and Appendix C
------------------------
The optimal-portfolio calculations are done in the notebooks in Notebooks/. Run them from that folder, in this order:

1. optimalportfolio_finitemat_nominal_baseline_final.ipynb produces all tables and figures in Section 5 of the main text and Appendix C except Figures 10, 16 and 17. It also writes Loadings_finer.csv, which the next notebook reads.
2. optimalportfolio_finitemat_nominal_factormimicking_final.ipynb produces Figure 10.
3. optimalportfolio_finitemat_nominal_ABN.ipynb produces the model-side moments of Table 3 from the simulated series in Data/ (see Section 6 below).
4. optimalportfolio_infinitemat_nominal_heteroskedastic_final.ipynb produces Figures 16 and 17.

Figures and CSV files are written to a folder named dump/ one level above Notebooks/ (created automatically) and to the Notebooks/ folder itself. The files in Manuscript/figdata are the versions used in the paper.

Section 6
---------
The ABN model is solved using the Julia code in ABN/.

- Figure 5 is produced by main_ABN_Nshocks.jl.
- The calculations for the 2 x 2 model are produced by main_ABN_2shocks.jl.
- Table 3 uses the simulated series Data/Sim_data.csv and Data/Sim_other_series.csv, which are processed by optimalportfolio_finitemat_nominal_ABN.ipynb. To regenerate them, run ABN/SimulateABN_2022/SimulateABN.jl from inside that subfolder (julia --project=.. SimulateABN.jl) and copy the two CSV files to Data/. The subfolder contains the version of the model code used for that simulation.

Note on data not included
-------------------------
The notebooks optimalportfolio_finitemat_nominal_ABN.ipynb and optimalportfolio_infinitemat_nominal_heteroskedastic_final.ipynb read Data/crsp_return_data.csv, which is not included because CRSP data cannot be redistributed. It is a monthly file from CRSP (Jan 1950 to Dec 2020) with columns DATE, vwretd, vwretx, ewretd, ewretx, sprtrn, spindx, totval, totcnt, usdval, usdcnt; only the S&P 500 return column "sprtrn" is used. Users with CRSP access can download the same series and place the file in Data/.
