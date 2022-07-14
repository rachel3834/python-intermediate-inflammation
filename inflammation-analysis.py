#!/usr/bin/env python3
"""Software for managing and analysing patients' inflammation data in our imaginary hospital."""

import argparse

from inflammation import models, views, serializers


def main(args):
    """The MVC Controller of the patient inflammation data system.

    The Controller is responsible for:
    - selecting the necessary models and views for the current task
    - passing data between models and views
    """
    in_files = args.infiles
    if not isinstance(in_files, list):
        in_files = [args.infiles]

    # If option is to serialize all of the data, initialize data list
    if args.view == 'serialize_all_patient_data':
        patients_info = []

    for filename in in_files:
        inflammation_data = models.load_csv(filename)

        if args.view == 'visualize':
            view_data = {
                        'average': models.daily_mean(inflammation_data),
                         'max': models.daily_max(inflammation_data),
                         'min': models.daily_min(inflammation_data)
                        }

            views.visualize(view_data)

        elif args.view == 'record':
            patient_data = inflammation_data[args.patient]
            observations = [models.Observation(day, value) for day, value in enumerate(patient_data)]
            patient = models.Patient('UNKNOWN', observations)

            views.display_patient_record(patient)

        elif args.view == 'serialize_patient_data':
            patient_data = inflammation_data[args.patient]
            observations = [models.Observation(day, value) for day, value in enumerate(patient_data)]
            patient = models.Patient('UNKNOWN', observations)

            out_file = filename.replace('.csv', '.json')
            views.serialize_patient_record(patient, out_file)

    if args.view == 'serialize_all_patient_data':
        serializers.PatientJSONSerializer.save(patients_info, 'data/patients.json')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='A basic patient inflammation data management system')

    parser.add_argument(
        'infiles',
        nargs='+',
        help='Input CSV(s) containing inflammation series for each patient')

    parser.add_argument(
        '--view',
        default='visualize',
        choices=['visualize', 'record', 'serialize_patient_data', 'serialize_all_patient_data'],
        help='Which view should be used?'
    )

    parser.add_argument(
        '--patient',
        type=int,
        default=0,
        help='Which patient should be displayed?'
    )

    args = parser.parse_args()

    main(args)
