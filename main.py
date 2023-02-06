from threading import Thread, Lock
import matplotlib.pyplot as plt
import numpy as np
import pyaudio

### WARNING: This code produces a loud, high-pitched tone. Turn down the volume! ###

class Generator(Thread):
    def __init__(self, amp: float, freq: float, offset: float, time: np.ndarray, type: str = 'sine'):
        super().__init__()

        self.signal_lock = Lock()

        self.amp_lock = Lock()
        self._amp = amp
        self.freq_lock = Lock()
        self._freq = freq
        self.offset_lock = Lock()
        self._offset = offset

        self.time_lock = Lock()
        self._time = time

        self.type_lock = Lock()
        self._type = type

        self._factor = 2 * np.pi * self.freq

    def sinewave(self):
        return self.amp * np.sin((self._factor * self.time) + self.offset).astype(np.float32)

    def squarewave(self):
        factor = 2 * np.pi * self.freq
        return self.amp * np.sign(np.sin((self._factor * self.time) + self.offset)).astype(np.float32)

    def sawtoothwave(self):
        return self.amp * (self._factor * self.time + self.offset) % (2 * np.pi) / np.pi - 1
    @property
    def signal(self):
        with self.signal_lock:
            if self.type == 'sine':
                return self.sinewave()
            elif self.type == 'square':
                return self.squarewave()
            elif self.type == 'sawtooth':
                return self.sawtoothwave()

    @property
    def type(self):
        with self.type_lock:
            return self._type
    @type.setter
    def type(self, value):
        with self.type_lock:
            self._type = value

    @property
    def amp(self):
        with self.amp_lock:
            return self._amp
    @amp.setter
    def amp(self, value):
        with self.amp_lock:
            self._amp = value

    @property
    def freq(self):
        with self.freq_lock:
            return self._freq
    @freq.setter
    def freq(self, value):
        with self.freq_lock:
            self._freq = value

    @property
    def offset(self):
        with self.offset_lock:
            return self._offset
    @offset.setter
    def offset(self, value):
        with self.offset_lock:
            self._offset = value

    @property
    def time(self):
        with self.time_lock:
            return self._time
    @time.setter
    def time(self, value):
        with self.time_lock:
            self._time = value

    def run(self):
        return self.signal
        pass

class Playback(Thread):
    def __init__(self, rate):
        super().__init__()

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=1, 
                                  rate=int(rate), 
                                  output=True)

        self.signal_lock = Lock()
        self._signal = None

    @property
    def signal(self):
        with self.signal_lock:
            return self._signal
    @signal.setter
    def signal(self, value):
        with self.signal_lock:
            self._signal = value

    def run(self):
        self.stream.start_stream()

    def play(self):
        if self.signal is None:
            pass
        else:
            self.stream.write(self.signal)

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        return

class Evaluator(Thread):
    def __init__(self, x, y, z, axis=0, matrix=np.identity(3)):
        super().__init__()

        self.signal_lock = Lock()

        self.transform_matrix_lock = Lock()
        self._matrix = matrix

        self._x = x
        self._y = y
        self._z = z

        self.axis_lock = Lock()
        self.axis = axis

        self.iterator = range(self.length)

    @property
    def length(self):
        return len(self.x.signal)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def matrix(self):
        with self.transform_matrix_lock:
            return self._matrix
    @matrix.setter
    def matrix(self, value):
        with self.transform_matrix_lock:
            self._matrix = value

    @property
    def axis(self):
        with self.axis_lock:
            return self._axis
    @axis.setter
    def axis(self, value):
        with self.axis_lock:
            self._axis = value

    @property
    def signal(self):
        with self.signal_lock:
            # get each element of x.signal, y.signal, z.signal and compose them into an array of vectors
            vector_array = np.array([self.x.signal, self.y.signal, self.z.signal]).T
            vector_array = np.matmul(vector_array, self.matrix)
            result = vector_array[:, self.axis].astype(np.float32)
            if np.max(result) == 0 or np.max(result) <= 1:
                return result
            else:
                return (result - np.min(result)) / (np.max(result) - np.min(result))
            
    def playback_callback(self, in_data, frame_count, time_info, status):
        return self.signal, pyaudio.paContinue

    def run(self):
        return self.signal
        pass


class Plotter(Thread):
    def __init__(self, data, matplotter, oneshot=False):
        super().__init__()
        self._matplotter = matplotter
        self._subplot = self._matplotter.subplots()
        self._oneshot = oneshot

        self.data_lock = Lock()
        self._data = data

    @property
    def data(self):
        with self.data_lock:
            return self._data

    @data.setter
    def data(self, value):
        with self.data_lock:
            self._data = value

    @property
    def fig(self):
        return self._subplot[0]

    @property
    def ax(self):
        return self._subplot[1]

    @property
    def line(self):
        x = np.arange(len(self.data))
        y = self.data
        return self.ax.plot(x, y)[0]

    def draw(self):
        self.line.set_ydata(self.data)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if not self._oneshot:
            self.clear()

    def clear(self):
        self.ax.clear()

    def run(self):
        self.draw()

rate = 44.1 * 1000
window = 1
time = np.arange(0, window, 1/rate).astype(np.float32)

playback = Playback(rate)
playback.start()

x = Generator(1, 440, 0, time, type='sawtooth')
y = Generator(1, 440, 0, time, type='sine')
z = Generator(1, 440, 0, time, type='square')
x.start()
y.start()
z.start()

evaluator = Evaluator(x, y, z, axis=0)
evaluator.start()

plotter = Plotter(evaluator.signal, plt, oneshot=True)
plotter.start()
plt.show(block=False)
plotter.clear()

# evaluator.join()
theta = 0
i = 0


while True:
    theta_x = np.radians(i)
    theta_y = np.radians(i)
    theta_z = np.radians(i)

    matrix_x = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    matrix_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
    matrix_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
    matrix = np.matmul(matrix_x, matrix_y)
    atrix = np.matmul(matrix, matrix_z)

    evaluator.matrix = matrix

    x.join()
    y.join()
    z.join()

    evaluator.join()

    plotter.data = evaluator.signal[:512]
    plotter.draw()


    playback.signal = evaluator.signal
    playback.play()
    playback.join()

    plotter.clear()
    plotter.join()

    #print(x.offset, y.offset, z.offset)

    i += 12.5

