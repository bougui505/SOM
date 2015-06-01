import numpy
import os


class Viewer:
    def __init__(self, som_object, output_js_data_file='data.js', output_html_data_file='heatmap.html'):
        # adapted from: http://blog.nextgenetics.net/?e=44
        #  output data for visualization in a browser with javascript/d3.js
        self.output_html_data_file = output_html_data_file
        self.output_js_data_file = output_js_data_file
        # unfold the SOM map and compute the U-matrix
        self.som = som_object
        self.som.graph.unfold_smap()
        self.input_matrix = self.som.graph.unfolded_umat
        representatives = self.som.get_representatives()
        self.representatives = self.som.graph.unfold_matrix(representatives)
        self.write_data_js_file()
        self.write_html()

    def write_data_js_file(self):
        matrix_output = []
        row = 0
        min_data_value = numpy.nanmin(self.input_matrix)
        max_data_value = numpy.nanmax(self.input_matrix)
        for rowData in self.input_matrix:
            col = 0
            row_output = []
            for colData in rowData:
                if not numpy.isnan(colData):
                    frame_id = self.representatives[row,col]
                    if numpy.isnan(frame_id):
                        frame_id = -1
                    else:
                        frame_id = numpy.int(frame_id)
                    row_output.append([colData, frame_id, row, col])
                else:
                    if min_data_value > 0:
                        none_value = 0
                    else:
                        none_value = min_data_value - min_data_value / 10
                    row_output.append([none_value, -1, row, col])
                col += 1
            matrix_output.append(row_output)
            row += 1

        shape = self.input_matrix.shape
        col_headers = range(shape[1])
        row_headers = range(shape[0])

        data_str = 'var maxData = ' + str(max_data_value) + ";\n" + \
                   'var minData = ' + str(min_data_value) + ";\n" + \
                   'var data = ' + str(matrix_output) + ";\n" + \
                   'var cols = ' + str(col_headers) + ";\n" + \
                   'var rows = ' + str([x for x in row_headers]) + ";"
        data_file = open(self.output_js_data_file, 'w')
        data_file.write(data_str)
        data_file.close()

    def write_html(self):
        script_dir = os.path.dirname(__file__)
        html_file = open('%s/heatmap.html'%script_dir, 'r')
        html_str = html_file.read()
        outfile = open(self.output_html_data_file, 'w')
        outfile.write(html_str)
        outfile.close()
