from inflammation import models, serializers

def test_patients_json_serializer():
    # Create some test data
    patients = [
        models.Patient('Alice', [models.Observation(i, i+1) for i in range(3)]),
        models.Patient('Bob', [models.Observation(i, 2 * i) for i in range(3)])
    ]

    # Save and reload the data
    output_file = 'patients.json'
    serializers.PatientJSONSerializer.save(patients, output_file)
    patients_new = serializers.PatientJSONSerializer.load(output_file)
    print(patients_new)

    # Check that we have the same data loaded
    for patient_new, patient in zip(patients_new, patients):
        assert patient_new == patient
