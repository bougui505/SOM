import numpy


class Viewer:
    def __init__(self, input_2d_matrix, output_js_data_file='data.js', output_html_data_file='heatmap.html'):
        # adapted from: http://blog.nextgenetics.net/?e=44
        #  output data for visualization in a browser with javascript/d3.js
        self.output_html_data_file = output_html_data_file
        self.output_js_data_file = output_js_data_file
        self.input_matrix = input_2d_matrix
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
                    row_output.append([colData, row, col])
                else:
                    if min_data_value > 0:
                        none_value = 0
                    else:
                        none_value = min_data_value - min_data_value / 10
                    row_output.append([none_value, row, col])
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
        html_str = """
<html>
   <head>
   <script type="text/javascript" src="http://d3js.org/d3.v3.js"></script>
   <script type="text/javascript" src="data.js"></script>
   <style>
      body {
         margin: 0px;
         padding: 0px;
         font: 12px Arial;
      }
   </style>
   </head>
   <body>
   <script type="text/javascript">
      //height of each row in the heatmap
      var h = 8;
      //width of each column in the heatmap
      var w = 8;

      //attach a SVG element to the document's body
      var mySVG = d3.select("body")
         .append("svg")
         .attr("width", (w * cols.length) + 400)
         .attr("height", (h * rows.length + 100))
         .style('position','absolute')
         .style('top',0)
         .style('left',0);

      //define a color scale using the min and max expression values
      var colorScale = d3.scale.linear()
        .domain([minData, (minData + maxData)/2, maxData])
        .range(["blue", "white", "red"]);

      //generate heatmap rows
      var heatmapRow = mySVG.selectAll(".heatmap")
         .data(data)
         .enter().append("g");

      //generate heatmap columns
      var heatmapRects = heatmapRow
         .selectAll(".rect")
         .data(function(d) {
            return d;
         }).enter().append("svg:rect")
         .attr('width',w)
         .attr('height',h)
         .attr('x', function(d) {
            return (d[2] * w) + 25;
         })
         .attr('y', function(d) {
            return (d[1] * h) + 50;
         })
         .style('fill',function(d) {
            if ( d[0] < minData - minData / 10 )
                {return "white"}
            else
                {return colorScale(d[0])};
         });

      //label columns
      var columnLabel = mySVG.selectAll(".colLabel")
         .data(cols)
         .enter().append('svg:text')
         .attr('x', function(d,i) {
            return ((i + 0.5) * w) + 25;
         })
         //.attr('y', 30)
         //.attr('class','label')
         //.style('text-anchor','middle')
         //.text(function(d) {return d;});

      //expression value label
      var expLab = d3.select("body")
         .append('div')
         .style('height',23)
         .style('position','absolute')
         .style('background','FFE53B')
         .style('opacity',0.8)
         .style('top',0)
         .style('padding',10)
         .style('left',40)
         .style('display','none');

      //heatmap mouse events
      heatmapRects
         .on('mouseover', function(d,i,j) {
            d3.select(this)
               .attr('stroke-width',1)
               .attr('stroke','black')

            output = '<b>' +
            '(' + rows[j] + ',' + cols[i] + ')' +
            ': ' + d3.round(data[j][i][0], 2) +
            '</b>';
            //for (var j = 0 , count = data[i].length; j < count; j ++ ) {
            //   output += data[i][j][0] + ", ";
            //}
            expLab
               .style('top',(rows[j] * h))
               .style('left', (cols[i] * w))
               .style('display','block')
               .html(output.substring(0,output.length));
      })
      .on('mouseout', function(d,i) {
         d3.select(this)
            .attr('stroke-width',0)
            .attr('stroke','none')
         expLab
            .style('display','none')
      });
   </script>
   </body>
</html>
        """
        outfile = open(self.output_html_data_file, 'w')
        outfile.write(html_str)
        outfile.close()
