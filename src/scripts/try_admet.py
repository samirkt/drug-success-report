from admet_ai import ADMETModel
model = ADMETModel()
test_pred = model.predict(smiles=['CCO'])  # ethanol
print(test_pred.columns)  # should show ~40 ADMET property columns
