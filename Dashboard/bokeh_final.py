import pandas as pd
import os
import numpy as np
from PIL import Image
from bokeh.layouts import widgetbox, column, row
from bokeh.plotting import figure, curdoc
from bokeh.palettes import Category20
from bokeh.core.properties import value
from bokeh.models import RangeSlider, RadioButtonGroup, MultiSelect, RadioGroup, ColumnDataSource, Slider, Select, Button, Range1d
from bokeh.models.widgets import Panel, Tabs, PreText

#### directory ####
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(BASE_DIR, "data/")
os.chdir(path)


############# load datasets and images #############
# load clean map
# Open image, and make sure it's RGB*A*
lena_img = Image.open('Lekagul_very_neat.png').convert('RGBA')
xdim, ydim = lena_img.size
# Create an array representation for the image `img`, and an 8-bit "4
# layer/RGBA" version of it `view`.
img = np.empty((ydim, xdim), dtype=np.uint32)
img = np.empty((ydim, xdim), dtype=np.uint32)
view = img.view(dtype=np.uint8).reshape((ydim, xdim, 4))
# Copy the RGBA image into view, flipping it so it comes right-side up
# with a lower-left origin
view[:,:,:] = np.flipud(np.asarray(lena_img))

# load main dataset
birds = pd.read_csv('birds_clean.csv')
test_birds = pd.read_csv('Test Birds Location.csv')

## create a color column in the dataframe associated to each specy
name_birds = birds['English_name'].unique()
num_birds = len(birds['English_name'].unique())
dict_colors = dict(zip(name_birds, Category20[num_birds]))

birds['colors'] = birds['English_name'].apply(lambda x: dict_colors[x])

TOOLS = "hover,zoom_in,zoom_out,box_zoom"



###### selection of representative signals  #####
def in_name(x, list_input):
    return (x in list_input)

dict_english_name_file_id = dict()
for name in birds.English_name.unique():
    dict_english_name_file_id[birds.loc[(birds['English_name']==name)&(birds['Quality']=='A'),'File ID'].iloc[0]] = name

#datasets for amplitude and signal plots
dict_df_signals = {}
dict_df_spectrum = {}
list_file_id = list(dict_english_name_file_id.keys())
for file_id in list_file_id:
    dict_df_signals[dict_english_name_file_id[file_id]] =pd.read_csv(r'amplitude_dataframe_%s'%dict_english_name_file_id[file_id])
    dict_df_spectrum[dict_english_name_file_id[file_id]]= pd.read_csv(r'spect_dataframe_%s' %dict_english_name_file_id[file_id])

list_df_test_signals = []
list_df_test_spectrum = []
for i in range(15):
    list_df_test_signals.append(pd.read_csv(r'amplitude_dataframe_%s' %i))
    list_df_test_spectrum.append(pd.read_csv(r'spect_dataframe_%s' % i))


# function to plot spectrum and amplitude plots
def create_signal_plot(data, title, x_label="Temps (secondes)", y_label="Amplitude", x_name='Time', y_name='Amplitude'):
    p = figure(plot_width=600, plot_height=150, output_backend="webgl", lod_threshold = 100, title=title, tools=TOOLS)
    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = y_label
    source = ColumnDataSource(data=data)
    p.line(x_name, y_name, source=source)
    return p


# datasets for stacked bar charts
dict_stacked_df_percentage = {}
dict_stacked_df_count = {}
for name in name_birds:
    dict_stacked_df_percentage[name] = pd.read_csv('stacked_df_percentage_%s' % name)
    dict_stacked_df_count[name] = pd.read_csv('stacked_df_count_%s' % name)



# prediction
array_pred = np.array([[0.05, 0.04, 0.08, 0.05, 0.03, 0.05, 0.04, 0.04, 0.03, 0.09, 0.12,
        0.05, 0.03, 0.03, 0.04, 0.07, 0.08, 0.07, 0.03],
       [0.02, 0.03, 0.05, 0.03, 0.06, 0.05, 0.06, 0.03, 0.03, 0.11, 0.07,
        0.05, 0.03, 0.05, 0.03, 0.1 , 0.13, 0.04, 0.04],
       [0.03, 0.02, 0.07, 0.05, 0.03, 0.06, 0.13, 0.05, 0.03, 0.07, 0.08,
        0.04, 0.02, 0.02, 0.03, 0.08, 0.12, 0.03, 0.04],
       [0.02, 0.02, 0.06, 0.03, 0.07, 0.06, 0.03, 0.03, 0.03, 0.08, 0.16,
        0.05, 0.02, 0.02, 0.02, 0.07, 0.17, 0.04, 0.03],
       [0.02, 0.02, 0.04, 0.03, 0.02, 0.05, 0.24, 0.08, 0.01, 0.05, 0.09,
        0.04, 0.01, 0.01, 0.04, 0.1 , 0.09, 0.02, 0.03],
       [0.03, 0.03, 0.04, 0.03, 0.02, 0.06, 0.23, 0.08, 0.02, 0.05, 0.07,
        0.04, 0.02, 0.01, 0.04, 0.12, 0.08, 0.02, 0.03],
       [0.05, 0.03, 0.09, 0.04, 0.04, 0.05, 0.04, 0.04, 0.05, 0.05, 0.11,
        0.03, 0.04, 0.02, 0.03, 0.09, 0.1 , 0.05, 0.04],
       [0.02, 0.01, 0.06, 0.03, 0.03, 0.04, 0.24, 0.05, 0.02, 0.04, 0.05,
        0.05, 0.01, 0.02, 0.03, 0.13, 0.12, 0.02, 0.03],
       [0.03, 0.03, 0.04, 0.03, 0.03, 0.05, 0.24, 0.09, 0.02, 0.05, 0.07,
        0.03, 0.01, 0.01, 0.04, 0.08, 0.09, 0.03, 0.02],
       [0.04, 0.02, 0.06, 0.03, 0.04, 0.07, 0.12, 0.04, 0.03, 0.06, 0.06,
        0.07, 0.02, 0.03, 0.05, 0.07, 0.11, 0.02, 0.04],
       [0.04, 0.02, 0.06, 0.03, 0.03, 0.06, 0.05, 0.04, 0.04, 0.09, 0.08,
        0.04, 0.02, 0.03, 0.03, 0.08, 0.18, 0.03, 0.04],
       [0.08, 0.04, 0.1 , 0.04, 0.04, 0.04, 0.03, 0.04, 0.06, 0.06, 0.1 ,
        0.04, 0.05, 0.02, 0.02, 0.08, 0.06, 0.05, 0.04],
       [0.03, 0.05, 0.06, 0.05, 0.05, 0.06, 0.02, 0.04, 0.06, 0.13, 0.08,
        0.04, 0.02, 0.03, 0.04, 0.07, 0.08, 0.05, 0.04],
       [0.03, 0.02, 0.05, 0.02, 0.05, 0.04, 0.05, 0.04, 0.05, 0.19, 0.06,
        0.05, 0.03, 0.03, 0.03, 0.07, 0.11, 0.03, 0.03],
       [0.03, 0.02, 0.04, 0.03, 0.03, 0.05, 0.22, 0.08, 0.02, 0.06, 0.07,
        0.03, 0.01, 0.02, 0.04, 0.08, 0.13, 0.02, 0.03]])
list_ordered_name = sorted(name_birds)

def modify_doc(doc):


    ####### plot tab3 (birds location) #####
    fig = figure(title="Birds recording in Lekagul Roadways",
                 x_range=(0, 250), y_range=(0, 200),
                 tools=TOOLS, width=900
                 )

    fig.xaxis.visible = False
    fig.yaxis.visible = False
    # Display the 32-bit RGBA image
    fig.image_rgba(image=[img], x=0, y=0, dw=200, dh=200)
    name_birds = birds['English_name'].unique()
    source = ColumnDataSource(birds)
    fig.circle('X', 'Y', source=source, size=8, color='colors', fill_alpha=0.7, legend='English_name')
    fig.asterisk(test_birds[' X'].tolist(), test_birds[' Y'].tolist(), size=20, line_color="black", line_width=3,
                 fill_color="blue", fill_alpha=0.5)
    rangeslider = RangeSlider(title="Date Range", start=1983, end=2018, value=(1983, 2018), step=1,
                              callback_policy='mouseup')
    multiselect = MultiSelect(title="Select your species : ", value=list(name_birds), options=list(name_birds))

    def callback(attr, old, new):
        date_tuple = rangeslider.value
        birds_name = multiselect.value
        dataset = birds[(birds['year'] <= date_tuple[1]) & (birds['year'] >= date_tuple[0]) & (
            birds['English_name'].apply(lambda x: in_name(x, birds_name)))]
        new_data = ColumnDataSource(dataset)
        source.data = new_data.data

    multiselect.on_change('value', callback)
    rangeslider.on_change('value', callback)
    inputs = widgetbox(rangeslider, multiselect)



    ####### main tab ######
    fig_1 = figure(title="Birds recording in Lekagul Roadways",
                 x_range=(0, 200), y_range=(0, 200),
                 tools=TOOLS, width=600, height=500
                 )
    fig_1.xaxis.visible = False
    fig_1.yaxis.visible = False
    fig_1.image_rgba(image=[img], x=0, y=0, dw=200, dh=200)
    source_1 = ColumnDataSource(birds)
    source_2 = ColumnDataSource(test_birds)
    fig_1.circle_cross(x=148, y=160, color='black', size=30, fill_alpha=0.3)
    fig_1.circle('X', 'Y', source=source_1, size=8, color='colors', fill_alpha=0.7)
    fig_1.asterisk(' X', ' Y', source=source_2, size=20, line_color="black", line_width=3,
                 fill_color="blue", fill_alpha=0.5)
    radio_button_group_test = RadioButtonGroup(
        labels=list(test_birds.ID.apply(str)), active=None, width=1300)
    rangeslider_2 = RangeSlider(title="Date Range", start=1983, end=2018, value=(1983, 2018), step=1,
                              callback_policy='mouseup')
    button_pred = Button(label='Predicted bird', button_type='danger')
    def update_1(attr, old, new):
        test_bird_num = radio_button_group_test.active
        if test_bird_num is not None:
            #button_pred.label=str(list_prediction[test_bird_num])
            dataset = test_birds[test_birds['ID']==test_bird_num+1]
            new_data = ColumnDataSource(dataset)
            source_2.data = new_data.data
            button_pred.label = list_ordered_name[np.argmax(array_pred[test_bird_num, :])]
            list_map = list(map(lambda x, y: x + ' ------ ' + str(y), list_ordered_name, array_pred[test_bird_num, :]))
            list_change = []
            for name_bird in name_birds:
                list_change.append(list_map[list_ordered_name.index(name_bird)])
            radio_button_group_train._property_values['labels'][:] = list_change
            plot_amplitude_test = create_signal_plot(list_df_test_signals[test_bird_num], 'test %s'%(test_bird_num +1))
            plot_amplitude_test.x_range = Range1d(0, 60)
            plot_spectrum_test = create_signal_plot(list_df_test_spectrum[test_bird_num], 'test %s'%(test_bird_num +1), "Fréquence", "Magnitude",
                                                    "Freq", "Magnitude")
            plot_spectrum_test.x_range = Range1d(0,0.02)
            signals_plot.children[0] = plot_amplitude_test
            signals_plot.children[2] = plot_spectrum_test
    radio_button_group_test.on_change('active', update_1)

    radio_button_group_train = RadioGroup(
        labels=list(name_birds), active=0)

    def update_2(attr, old, new):
        idx_name_bird = radio_button_group_train.active
        date_tuple = rangeslider_2.value
        if idx_name_bird is not None:
            dataset = birds[(birds['year'] <= date_tuple[1]) & (birds['year'] >= date_tuple[0]) &
                            (birds['English_name'] == name_birds[idx_name_bird])]
            new_data = ColumnDataSource(dataset)
            source_1.data = new_data.data
            if new in range(15):
                plot_amplitude_test = create_signal_plot(dict_df_signals[name_birds[idx_name_bird]], name_birds[idx_name_bird])
                plot_amplitude_test.x_range = Range1d(0, 60)
                plot_spectrum_test = create_signal_plot(dict_df_spectrum[name_birds[idx_name_bird]], name_birds[idx_name_bird], "Fréquence", "Magnitude",
                                                        "Freq", "Magnitude")
                plot_spectrum_test.x_range = Range1d(0, 0.02)
                signals_plot.children[1] = plot_amplitude_test
                signals_plot.children[3] = plot_spectrum_test
        else:
            pass

    radio_button_group_train.on_change('active', update_2)
    rangeslider_2.on_change('value', update_2)
    plot_amplitude_train = create_signal_plot(dict_df_signals[name_birds[0]], name_birds[0])
    plot_amplitude_train.x_range = Range1d(0, 60)
    plot_amplitude_test = create_signal_plot(list_df_test_signals[0], 'test 1')
    plot_amplitude_test.x_range = Range1d(0, 60)
    plot_spectrum_train = create_signal_plot(dict_df_spectrum[name_birds[0]], name_birds[0], "Fréquence", "Magnitude", "Freq", "Magnitude")
    plot_spectrum_train.x_range = Range1d(0, 0.02)
    plot_spectrum_test = create_signal_plot(list_df_test_spectrum[0], 'test 1', "Fréquence", "Magnitude",
                                                                    "Freq", "Magnitude")
    plot_spectrum_test.x_range = Range1d(0, 0.02)
    signals_plot = column([plot_amplitude_test, plot_amplitude_train, plot_spectrum_test, plot_spectrum_train])
    row_map_signal = row([radio_button_group_train, column([fig_1, rangeslider_2]), signals_plot])
    column_main = column([row([widgetbox([PreText(text="Prediction : "),radio_button_group_test]), button_pred]),
                          row_map_signal])
    tab1 = Panel(child=column_main,title="Main view")
    tab2 = Panel(child=column(inputs, fig), title="Birds location")




    ##### tab2 :  birds call /song evolution
    birds_call_song = birds.copy()
    birds_call_song['Vocalization_type'] = birds_call_song['Vocalization_type'].apply(lambda x: x.lower().strip())
    birds_call_song.dropna(inplace=True)
    fig_3 = figure(title="Birds recording in Lekagul Roadways",
                 x_range=(0, 250), y_range=(0, 200),
                 tools=TOOLS, width=900
                 )

    fig_3.xaxis.visible = False
    fig_3.yaxis.visible = False
    fig_3.image_rgba(image=[img], x=0, y=0, dw=200, dh=200)
    birds_call = birds_call_song[birds_call_song['Vocalization_type']=='call']
    birds_song = birds_call_song[birds_call_song['Vocalization_type'] == 'song']
    birds_call_song = birds_call_song[birds_call_song['Vocalization_type'] == 'call, song']

    source_call = ColumnDataSource(birds_call)
    source_song = ColumnDataSource(birds_song)
    source_call_song = ColumnDataSource(birds_call_song)
    fig_3.circle_cross(x=148, y=160, color='black', size=30, fill_alpha=0.3)
    fig_3.circle('X', 'Y', source=source_call, size=16, color='colors', fill_alpha=0.7, legend='Vocalization_type')
    fig_3.triangle('X', 'Y', source=source_song, size=16, color='colors', fill_alpha=0.7, legend='Vocalization_type')
    fig_3.square('X', 'Y', source=source_call_song, size=16, color='colors', fill_alpha=0.7, legend='Vocalization_type')
    fig_3.asterisk(test_birds[' X'].tolist(), test_birds[' Y'].tolist(), size=20, line_color="black", line_width=3,
                 fill_color="blue", fill_alpha=0.5)
    slider = Slider(title="Select the year:", start=1983, end=2018, value=2017, step=1)
    select = Select(title="Select your specie : ", value='Rose-crested Blue Pipit', options=list(name_birds))
    stacked_bar = figure(plot_height=250, title="Vocalization type percentage by year",
                         toolbar_location=None)

    source_stacked_bar = ColumnDataSource(dict_stacked_df_percentage['Rose-crested Blue Pipit'])
    stacked_bar.vbar_stack(['call', 'song'], x='Year', width=0.9,
                           source=source_stacked_bar, color=['silver', 'lightblue'],
                           legend=[value(x) for x in ['call', 'song']])
    stacked_bar.legend.location = "bottom_left"
    stacked_bar_count = figure(plot_height=250, title="Vocalization type count by year",
                         toolbar_location=None)

    source_stacked_bar_count = ColumnDataSource(dict_stacked_df_count['Rose-crested Blue Pipit'])
    stacked_bar_count.vbar_stack(['call', 'song'], x='Year', width=0.9,
                           source= source_stacked_bar_count, color=['silver', 'lightblue'],
                           legend=[value(x) for x in ['call', 'song']])
    stacked_bar_count.legend.location = "bottom_left"

    def callback_call(attr, old, new):
        year = slider.value
        birds_name = select.value
        dataset_call = birds_call[(birds_call['year'] == year)&(birds_call['English_name'] == birds_name)]
        new_data = ColumnDataSource(dataset_call)
        source_call.data = new_data.data
        dataset_song = birds_song[(birds_song['year'] == year) & (birds_song['English_name'] == birds_name)]
        new_data = ColumnDataSource(dataset_song)
        source_song.data = new_data.data
        dataset_call_song = birds_call_song[(birds_call_song['year'] == year) & (birds_call_song['English_name'] == birds_name)]
        new_data = ColumnDataSource(dataset_call_song)
        source_call_song.data = new_data.data
        new_data = ColumnDataSource(dict_stacked_df_percentage[birds_name])
        source_stacked_bar.data = new_data.data
        new_data = ColumnDataSource(dict_stacked_df_count[birds_name])
        source_stacked_bar_count.data = new_data.data

    select.on_change('value', callback_call)
    slider.on_change('value', callback_call)
    tab3 = Panel(child=column(select, row([fig_3,column(stacked_bar,stacked_bar_count)]), slider), title="Birds Evolution")

    #### regroup the tabs into one dashboard #####
    tabs = Tabs(tabs=[tab1, tab3, tab2])
    doc.add_root(tabs)

doc = curdoc()
doc = modify_doc(doc)




