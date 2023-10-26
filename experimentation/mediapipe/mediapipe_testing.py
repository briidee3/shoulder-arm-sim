# BD 2023
# This file used for testing mediapipe functionality for use in simulating forces pertaining to the human arm system
# as well as implementing and integrating a custom algorithm for discerning depth from 2D skeleton data


import mediapipe as mp
import numpy as np
import cv2


# good resource: https://developers.google.com/mediapipe/framework/getting_started/python_framework

# initialization of graph
config_text = """
    input_stream: 'in_stream'
    output_stream: 'out_stream'
    node {
        calculator: 'PassThroughCalculator'
        input_stream: 'in_stream'
        output_stream: 'out_stream'
    }
"""
graph = mp.CalculatorGraph(graph_config = config_text)
output_packets = []
graph.observe_output_stream(
    'out_stream',
    lambda stream_name, packet:
    output_packets.append(mp.packet_getter.get_str(packet))
)


# run graph
graph.start_run()

graph.add_packet_to_input_stream('in_stream', mp.packet_creator.create_string('abc').at(0))

rgb_img = cv2.cvtColor(cv2.imread('test_image.jpg'), cv2.COLOR_BGR2RGB)
graph.add_packet_to_input_stream('in_stream', mp.packet_creator.create_image_frame(image_format = mp.ImageFormat.SRGB, data = rgb_img).at(1))


# shutdown graph
graph.close()
