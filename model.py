import numpy as np
import matplotlib.pyplot as plt

dt              = 0.01              # timestep (s)
v_reset         = 0                 # reset voltage
v_threshold     = 1                 # spike threshold
tau             = 1                 # synaptic time constant
refractory_time = np.ceil(0.05/dt)   # refractory time (timesteps)
u_window        = 0.1               # RDD input drive window

# INPUT PARAMETERS
tau_s = 0.003/dt # synaptic time constant
tau_L = 0.01/dt  # leak time constant
mem   = int(20/dt)

def kappa(x):
    return (np.exp(-x/tau_L) - np.exp(-x/tau_s))/(tau_L - tau_s)

def get_kappas(n=mem):
    return np.array([kappa(i+1) for i in range(n)])

kappas = np.flipud(get_kappas(mem))[:, np.newaxis] # initialize kappas array

class Layer():
	def __init__(self, size):
		self.size                 = size
		self.u                    = v_reset*np.ones((self.size, 1))
		self.v                    = v_reset*np.ones((self.size, 1))
		self.dv_dt                = np.zeros((self.size, 1))
		self.fired                = np.zeros((self.size, 1)).astype(bool)
		self.refractory_time_left = np.zeros((self.size, 1))
		self.max_u                = np.zeros((self.size, 1))
		self.n_spikes             = np.zeros((self.size, 1))
		self.spike_hist           = np.zeros((self.size, mem), dtype=np.int8)
		self.feedback             = np.zeros((self.size, 1))

	def update(self, I, reset_u=False, feedback=None):
		if reset_u:
			# reset input drive
			self.u                    = self.v.copy()
			self.max_u                = np.zeros((self.size, 1))
			self.n_spikes             = np.zeros((self.size, 1))
			self.feedback             = np.zeros((self.size, 1))

		# calculate change in voltage and input drive
		self.dv_dt    = -self.v/tau + I
		self.u        += dt*self.dv_dt
		self.v        += dt*self.dv_dt

		if feedback is not None:
			self.feedback += feedback

		self.max_u[self.u > self.max_u] = self.u[self.u > self.max_u]

		refractory_mask = self.refractory_time_left > 0
		threshold_mask  = self.v >= v_threshold

		# print("th", threshold_mask)
		# print("re", refractory_mask)

		# self.v[fired_mask] = v_reset
		self.fired[refractory_mask] = False

		self.fired[threshold_mask] = True
		self.n_spikes[threshold_mask] += 1
		self.refractory_time_left[threshold_mask] = refractory_time

		self.refractory_time_left[self.refractory_time_left > 0] -= 1
		fired_mask      = self.fired

		self.v[fired_mask] = v_reset

		# print(self.spike_hist[:, 1:].shape)
		# print(self.fired.astype(int).shape)
		# print(self.fired.astype(int))
		# print(self.fired)
		# update spike history
		self.spike_hist = np.concatenate([self.spike_hist[:, 1:], self.fired.astype(int)], axis=1)

class Neuron():
	def __init__(self):
		self.u                    = v_reset
		self.v                    = v_reset
		self.dv_dt                = 0
		self.fired                = False
		self.refractory_time_left = 0
		self.max_u                = 0
		self.n_spikes             = 0
		self.spike_hist           = np.zeros(mem, dtype=np.int8)
		self.feedback             = 0

	def update(self, I, reset_u=False, feedback=None):
		if reset_u:
			# reset input drive
			self.u        = self.v
			self.max_u    = 0
			self.n_spikes = 0
			self.feedback = 0

		# calculate change in voltage and input drive
		self.dv_dt    = -self.v/tau + I
		self.u        += dt*self.dv_dt
		self.v        += dt*self.dv_dt

		if feedback is not None:
			self.feedback += feedback

		if self.u > self.max_u:
			# update maximum input drive
			self.max_u = self.u

		if self.refractory_time_left > 0:
			# keep voltage at reset value
			self.v = v_reset

			self.fired = False

		if self.v >= v_threshold:
			# spike, update spike count and refractory time left

			self.fired = True

			self.n_spikes += 1

			self.refractory_time_left = refractory_time
		else:
			# decrement refractory time left
			self.refractory_time_left -= 1

		# update spike history
		self.spike_hist = np.concatenate([self.spike_hist[1:], np.ones(1)*int(self.fired)], axis=-1)

if __name__ == "__main__":
	n_trials = 500

	ws   = []
	vals = []

	layer_1_size = 1

	for k in range(n_trials):
		print("Trial {}/{}.".format(k+1, n_trials))
		# create 2 layers, 1 neuron each
		layer_1 = Layer(size=layer_1_size)
		layer_2 = Layer(size=1)

		# set weight from neuron 1 to neuron 2
		w = 100*np.random.uniform(-5, 5, size=(1, layer_1_size))

		# set total time of simulation (timesteps)
		total_time = int(100/dt)

		# set input drive calculation window (timesteps)
		window = 0.2/dt

		# create array of timesteps
		time_array = np.arange(total_time)

		# create independent inputs for neuron 1 and neuron 2
		input_1 = np.ones((layer_1_size, total_time))
		for j in range(layer_1_size):
			input_1[j] = 2*(np.random.normal(0, 1, size=total_time) + 0.5)

		input_2 = (np.random.normal(0, 1, size=total_time) + 0.5) + 2

		# print(input_1.shape)

		# initialize recording arrays
		voltages_1   = np.zeros((layer_1_size, total_time))
		voltages_2   = np.zeros((1, total_time))
		drives_1     = np.zeros((layer_1_size, total_time))
		drives_2     = np.zeros((1, total_time))
		feedback_1   = np.zeros((layer_1_size, total_time))
		below_us     = [ [] for j in range(layer_1_size) ]
		above_us     = [ [] for j in range(layer_1_size) ]
		below_spikes = [ [] for j in range(layer_1_size) ]
		above_spikes = [ [] for j in range(layer_1_size) ]

		print("Running simulation...")

		for t in range(total_time):
			# reset input drives if the window has ended
			if (t+1) % window == 0:
				reset_u = True
			else:
				reset_u = False

			if (t+1) % window == 0:
				for j in range(layer_1_size):
					# calculate RDD values
					# if np.abs(layer_1.max_u[j] - v_threshold) < u_window:
					if layer_1.max_u[j] < v_threshold:
						below_us[j].append(layer_1.max_u[j, 0])
						below_spikes[j].append(layer_1.feedback[j, 0])
					else:
						above_us[j].append(layer_1.max_u[j, 0])
						above_spikes[j].append(layer_1.feedback[j, 0])
			# print("shape", layer_2.spike_hist.shape)
			# update neuron activities
			# print(np.dot(layer_1.spike_hist, kappas))
			layer_1.update(input_1[:, t][:, np.newaxis], reset_u=reset_u, feedback=np.dot(layer_2.spike_hist, kappas))
			# layer_2.update(input_2[t], reset_u=reset_u)
			# print(layer_1.spike_hist)
			# print(layer_1.fired[0, 0], layer_1.refractory_time_left[0, 0])
			layer_2.update(np.dot(w, np.dot(layer_1.spike_hist, kappas)) + input_2[t], reset_u=reset_u)

			# record voltages and input drives
			voltages_1[:, t] = layer_1.v[:, 0]
			voltages_2[:, t] = layer_2.v[:, 0]
			drives_1[:, t]   = layer_1.u[:, 0]
			drives_2[:, t]   = layer_2.u[:, 0]
			feedback_1[:, t] = layer_1.feedback[:, 0]

		below_us     = np.array(below_us)
		above_us     = np.array(above_us)
		below_spikes = np.array(below_spikes)
		above_spikes = np.array(above_spikes)

		# rdd_value = [ np.mean(above_spikes[j]) - np.mean(below_spikes[j]) for j in range(layer_1_size) ]

		# print(below_spikes.shape)

		A = np.vstack([below_us, np.ones(below_us.shape)]).T
		# print(A.shape)
		m_below, c_below = np.linalg.lstsq(A, below_spikes.T)[0]
		x = np.linspace(v_threshold - u_window, v_threshold, 100)

		A = np.vstack([above_us, np.ones(above_us.shape)]).T
		m_above, c_above = np.linalg.lstsq(A, above_spikes.T)[0]
		x = np.linspace(v_threshold, v_threshold + u_window, 100)

		rdd_value = (m_above*v_threshold + c_above) - (m_below*v_threshold + c_below)

		if k == 0:
			plt.subplot(211)

			plt.plot(input_1[0], 'g')
			plt.plot(input_2, 'purple')
			plt.plot(voltages_1[0], 'r')
			plt.plot(voltages_2[0], 'b')
			plt.plot(drives_1[0], 'r', linestyle='--')
			plt.plot(drives_2[0], 'b', linestyle='--')
			plt.plot(time_array, v_threshold*np.ones(total_time), 'k', linestyle='--')
			plt.xlim(0, total_time)

			plt.xlabel("Time (timesteps)")

			plt.subplot(212)

			plt.scatter(below_us, below_spikes, c='b', alpha=0.5)
			plt.scatter(above_us, above_spikes, c='r', alpha=0.5)

			x = np.linspace(v_threshold - u_window, v_threshold, 100)
			plt.plot(x, m_below*x + c_below, 'b', linestyle='--')

			x = np.linspace(v_threshold, v_threshold + u_window, 100)
			plt.plot(x, m_above*x + c_above, 'r', linestyle='--')

			x = np.linspace(v_threshold - u_window, v_threshold, 100)
			plt.plot(x, np.mean(below_spikes)*np.ones(100), 'b', linestyle='--', alpha=0.5)

			x = np.linspace(v_threshold, v_threshold + u_window, 100)
			plt.plot(x, np.mean(above_spikes)*np.ones(100), 'r', linestyle='--', alpha=0.5)

			plt.axvline(x=v_threshold, color='k', linestyle='--')
			plt.xlim(v_threshold - u_window, v_threshold + u_window)

			plt.xlabel("Neuron 1 maximum input drive")
			plt.ylabel("Neuron 2 spiking activity")

			# rdd_value = (m_above*v_threshold + c_above) - (m_below*v_threshold + c_below)

			print("Neuron 1 -> 2 weight: {}.".format(w))
			print("RDD value: {}.".format(rdd_value))

			plt.show()

		ws.append(w)
		vals.append(rdd_value)

	# print([ w[0, 0] for w in ws ], vals[0])
	# print([ val[0] for val in vals ])
	# print(vals[0].shape)

	plt.scatter([ w[0, 0] for w in ws ], [ val[0] for val in vals ], c='g', alpha=0.5)
	plt.axvline(x=0, color='k', linestyle='--')
	plt.axhline(y=0, color='k', linestyle='--')
	plt.xlabel("Neuron 1 -> 2 Weight")
	plt.ylabel("RDD Value")
	plt.show()