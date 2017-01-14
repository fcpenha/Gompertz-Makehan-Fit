# Gompertz-Makehan-Fit

Gompertz–Makeham law of mortality fit to the 2016 English Life Table (ELT16)

**Check out our wiki for equations and graphical results.**

The ELT16 table can be found in the **data** directory, in the **xlsx format**. We read and interpret the table by using the following Python libraries:

    import pyexcel_xlsx as pe
    import json
    import ast

Usage:

    # Read data from ELT16.xlsx
    age = np.asarray([
                         n[0] for n in list(
            ast.literal_eval(
                json.dumps(
                    pe.get_data('./data/ELT16.xlsx', start_column=0, column_limit=1)
                )[:-2][101:]
            )
        )][:-5],
                     dtype=float)
