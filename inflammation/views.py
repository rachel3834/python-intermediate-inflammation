"""Module containing code for plotting inflammation data."""

from inflammation import serializers


def visualize(data_dict):
    """Display plots of basic statistical properties of the inflammation data.

    :param data_dict: Dictionary of name -> data to plot
    """
    # TODO(lesson-design) Extend to allow saving figure to file

    num_plots = len(data_dict)
    fig = plt.figure(figsize=((3 * num_plots) + 1, 3.0))

    for i, (name, data) in enumerate(data_dict.items()):
        axes = fig.add_subplot(1, num_plots, i + 1)

        axes.set_ylabel(name)
        axes.plot(data)

    fig.tight_layout()

    plt.show()


def display_patient_record(patient):
    """Display an individual patients information"""

    print(patient.name)
    if len(patient.observations) > 0:
        for datum in patient.observations:
            print(datum)
    else:
        print('No observations made so far')


def serialize_patient_record(patient, output_file):
    """Output patient data in JSON format"""

    serializers.PatientJSONSerializer.save([patient], output_file)