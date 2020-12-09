from bokeh.plotting import figure, output_file, show, Column, Row
from bokeh.models import DataTable, TableColumn, PointDrawTool, ColumnDataSource, Div, Button, Slider, HoverTool
from bokeh.io import curdoc
from bokeh.events import ButtonClick, MouseMove, PanEnd
from bokeh.models.renderers import GlyphRenderer

from lpsolve55 import lpsolve, EQ, GE, LE
import numpy as np
import itertools

def compute_network(source, floating=1., fixed=0.):
    # source is dict, keys 'x' and 'y' containing coordinates, 'flow' containing capacity
    # objective: return network to be constructed for which construction costs are minimal
    # while keeping flow constraints satisfied
    coords = list(zip(source.data['x'], source.data['y']))

    flows = {}
    for i, coord in enumerate(coords):
        flows[coord] = np.float(source.data['flow'][i])

    distances = []
    for start, end in itertools.permutations(coords, 2):
        # compute Euclidean distance
        distance = np.sqrt((end[0] - start[0])**2. + (end[1] - start[1])**2.)

        distances.append([start, end, distance])

    # decision variables: 
    # binary: construct pipe from point i to j for all points i to points j
    # flow vars: flow from point i to point j for all points i to points j
    # binary variables: only flow if pipe is constructed, from point i to point j for all points i to points j
    # plus one slack variable for overflow, one for underflow
    nrSlackVar = 0
    numBinaryVars = len(distances)
    numFlowVars = len(distances)
    fixedCostVar = 1

    nrDecVar = numBinaryVars + numFlowVars + numBinaryVars + fixedCostVar + nrSlackVar

    m = 0.00001
    M = 100000.

    lp = lpsolve('make_lp', 0, nrDecVar)
    lpsolve('set_minim', lp) 

    # construction costs scale with distance, as of now no fixed amount
    print(f"distances are {[floating * d[2] for d in distances] + [0.] * numFlowVars + [0.] * numBinaryVars + [1] + [1000.] * nrSlackVar}")
    ret = lpsolve('set_obj_fn', lp, [floating * d[2] for d in distances] + [0.] * numFlowVars + [0.] * numBinaryVars + [1] + [1000.] * nrSlackVar)

    # set binary vars
    for i in np.arange(1, numBinaryVars + 1):
        ret = lpsolve('set_binary', lp, i, True)

    for i in np.arange(numBinaryVars + numFlowVars + 1, numBinaryVars + numFlowVars + numBinaryVars + 1):
        ret = lpsolve('set_binary', lp, i, True)

    for i in np.arange(0, numFlowVars):
        # f_12 >= mq_12 - M(1-q_12)
        coeff = np.zeros(nrDecVar)

        coeff[numBinaryVars + i] = 1.
        coeff[numBinaryVars + numFlowVars + 1] = -(M + m)
        ret = lpsolve('add_constraint', lp, coeff, GE, -M)

        # f_12 <= Mq_12 - m(1-q_12)
        coeff = np.zeros(nrDecVar)

        coeff[numBinaryVars + i] = 1.
        coeff[numBinaryVars + numFlowVars + 1] = -(M + m)
        ret = lpsolve('add_constraint', lp, coeff, LE, -m)
    
    # ieder punt met nonzero flow moet tenminste 1 edge hebben
    for i, coord in enumerate(coords):
        flow = flows[coord]
        print(f"For coordinate {coord} the flow is {flow}")        

        coeff = np.zeros(nrDecVar)
        if flow != 0.:
            for j, (start, end, _) in enumerate(distances):        
                # find linked edges
                if((start == coord) | (end == coord)):
                    print(f"For coordinate {coord} we found matching capacity line {start} to {end}")
                    coeff[j] = 1.
            
            ret = lpsolve('add_constraint', lp, coeff, GE, 1)


    
    for i, coord in enumerate(coords):
        flow = flows[coord]
        print(f"For coordinate {coord} the flow is {flow}")

        coeff = np.zeros(nrDecVar)
        for j, (start, end, _) in enumerate(distances):
            # find linked edges
            if(start == coord):
                print(f"For coordinate {coord} we found matching capacity line {start} to {end}")
                coeff[numBinaryVars + j] = -1.
            elif (end == coord):
                print(f"For coordinate {coord} we found matching capacity line {start} to {end}")
                coeff[numBinaryVars + j] = 1.

        print(f"{coeff} = {flow}")
        ret = lpsolve('add_constraint', lp, coeff, EQ, flow)

    # implement fixed costs per pipeline
    coeff = np.zeros(nrDecVar)
    coeff[:numBinaryVars] = fixed
    coeff[numBinaryVars + numFlowVars + numBinaryVars] = -1.
    ret = lpsolve('add_constraint', lp, coeff, EQ, 0.)

    ret = lpsolve('write_lp', lp, 'a.lp')
    lpsolve('solve', lp)

    opt_result = {}
    status = lpsolve('get_status', lp)

    opt_result['statustext'] = lpsolve('get_statustext', lp, status)

    opt_result['lpvars'] = lpsolve('get_variables', lp)[0]
    if not isinstance(opt_result['lpvars'], list):
        opt_result['lpvars'] = [opt_result['lpvars']]

    opt_result['objfun'] = lpsolve('get_obj_fun', lp)[0]
    opt_result['objfunvalue'] = lpsolve('get_objective', lp)

    lpsolve('delete_lp', lp)

    text = f"Objective: minimize construction cost of network<p>"
    text += f"Construction cost is based on number of pipes and distance between nodes.<br>"
    text += f"Additional constraints imposed: flows in network must be balanced.<p>"

    text += f"Optimization result: {opt_result['statustext']}!<p>"
    text += f"Network total cost: {np.round(opt_result['objfunvalue'], 1)}<p>"

    text += f"Total cost due to distance covered with pipelines: {np.round(np.sum(np.asarray(opt_result['lpvars'][:numBinaryVars]) * np.asarray(opt_result['objfun'][:numBinaryVars])), 1)}<br>"
    text += f"Total cost due to number of pipelines: {np.round(np.sum(opt_result['lpvars'][numBinaryVars + numFlowVars + numBinaryVars]), 1)}<p>"

    # text += f"Obj fun {opt_result['objfun']}<p>"

    # text += f"LP vars: {opt_result['lpvars']}<p>"
    for i, coord in enumerate(coords):
        flow = flows[coord]

        total_in = 0.
        total_out = 0.
        for j, (start, end, _) in enumerate(distances):
            # find linked edges
            if(start == coord):
                total_in += opt_result['lpvars'][numBinaryVars + j]
            elif (end == coord):
                total_out += opt_result['lpvars'][numBinaryVars + j]

        balance = flow + total_in - total_out
        text += f"Node {coord} has flow {np.round(flow, 1)}, total inflow {np.round(total_in, 1)} and total outflow {np.round(total_out, 1)}, balance {np.round(balance, 1)}<p>"


    opt_result['edges'] = []
    for i, result_flow in enumerate(opt_result['lpvars'][numBinaryVars:numBinaryVars + numFlowVars]):
        fr = distances[i][0]
        to = distances[i][1]

        # text += f"From point {fr} to {to}: {np.round(result_flow, 1)}<p>"

        if result_flow != 0.:
            opt_result['edges'].append([fr, to, result_flow])

    # text += "</h2>"

    return opt_result, text


def draw_lines(p, source, opt_result):
    # first, remove old lines
    remove_glyphs(p, ['line'])

    for fr, to, result_flow in opt_result['edges']:
        x1 = fr[0]
        x2 = to[0]
        y1 = fr[1]
        y2 = to[1]

        if result_flow != 0.:
            p.line([x1, x2], [y1, y2], line_width=result_flow/5., name='line')
            print(f"Drawing line from {[x1, x2]} to {y1, y2} with {result_flow} capacity!")
    
    # hover_tool = HoverTool(mode='vline', names=['line'], tooltips=[("Built capacity", "@line_width"),])
    # p.add_tools(hover_tool)

def remove_glyphs(figure, glyph_name_list):
    renderers = figure.select(dict(type=GlyphRenderer))
    for r in renderers:
        if r.name in glyph_name_list:
            col = r.glyph.y
            r.data_source.data[col] = [np.nan] * len(r.data_source.data[col])

p = figure(x_range=(0, 10), y_range=(0, 10), tools=[],
           title='Draw points in the network')
p.background_fill_color = 'lightgrey'

source = ColumnDataSource({
    'x': [2, 8, 5], 'y': [2, 2, 6], 'flow': ['-5', '-5', '10']
    # 'x': [1, 9], 'y': [1, 9], 'flow': ['10', '-10']
})

renderer = p.scatter(x='x', y='y', source=source, color='blue', size=10)
columns = [TableColumn(field="x", title="x"),
           TableColumn(field="y", title="y"),
           TableColumn(field='flow', title='flow')]
table = DataTable(source=source, columns=columns, editable=True, height=200)

draw_tool = PointDrawTool(renderers=[renderer], empty_value='0')
p.add_tools(draw_tool)
p.toolbar.active_tap = draw_tool

textbox = Div(text="", width=200, height=100)
floating = 1.
fixed = 0.
# show(Column(p, table))
button = Button(label='Solve Network')
def button_click_event(event=None, floating=1., fixed=0.):
    opt_result, text = compute_network(source, floating=floating, fixed=fixed)

    textbox.text = text

    draw_lines(p, source, opt_result)

button.on_event(ButtonClick, button_click_event)





p.on_event(PanEnd, button_click_event)
lumpSumCost = Slider(title="Fixed cost pipe", value=0.0, start=0.0, end=500.0, step=50)
floatingCost = Slider(title="Floating cost pipe", value=1.0, start=0.0, end=5.0, step=0.5)


def update_data(attrname, old, new):
    fixed = lumpSumCost.value
    floating = floatingCost.value

    button_click_event(floating=floating, fixed=fixed)

for w in [lumpSumCost, floatingCost]:
    w.on_change('value', update_data)


curdoc().add_root(Row(Column(p, table, width=800), Column(lumpSumCost, floatingCost, button, textbox, width=300)))
curdoc().title = "Network"

