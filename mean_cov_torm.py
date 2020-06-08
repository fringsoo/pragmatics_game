import numpy as np
import scipy.stats as st

keys = ['baseline', 'prag_sample_0', 'prag_sample_0_5', 'prag_argmax', 'prag_argmax_virtual', 'rsa_2', 'ibr_2', 'rsa_3', 'ibr_3', 'rsa_equi', 'rsa_equi_virtual', 'ibr_equi', 'ibr_equi_virtual', 'table', 'table_virtual', 'table_special', 'table_special_virtual']
means_challenge = {}
mps_challenge = {}
css_challenge = {}
mps_challenge_real = {}
css_challenge_real = {}

means_challenge['baseline'] = np.array([0.518, 0.533, 0.569, 0.538, 0.490])
mps_challenge['baseline'] = np.array([0.294, 0.292, 0.312, 0.312, 0.263])
css_challenge['baseline'] = np.array([0.408, 0.429, 0.462, 0.428, 0.386])

means_challenge['prag_sample_0'] = np.array([0.538, 0.559, 0.596, 0.538, 0.510])
mps_challenge['prag_sample_0'] = np.array([0.130, 0.146, 0.135, 0.111, 0.114])
css_challenge['prag_sample_0'] = np.array([0.417, 0.433, 0.478, 0.428, 0.399])

means_challenge['prag_sample_0_5'] = np.array([0.482, 0.518, 0.547, 0.527, 0.420])
mps_challenge['prag_sample_0_5'] = np.array([0.271, 0.281, 0.304, 0.301, 0.221])
css_challenge['prag_sample_0_5'] = np.array([0.374, 0.396, 0.426, 0.400, 0.317])

means_challenge['prag_argmax'] = np.array([0.528, 0.559, 0.596, 0.566, 0.530])
mps_challenge['prag_argmax'] = np.array([0.295, 0.294, 0.313, 0.314, 0.266])
css_challenge['prag_argmax'] = np.array([0.413, 0.445, 0.479, 0.444, 0.413])

means_challenge['prag_argmax_virtual'] = np.array([0.518, 0.533, 0.564, 0.538, 0.495])
mps_challenge['prag_argmax_virtual'] = np.array([0.270, 0.274, 0.278, 0.287, 0.245])
css_challenge['prag_argmax_virtual'] = np.array([0.410, 0.429, 0.460, 0.428, 0.391])

means_challenge['rsa_2'] = np.array([0.538, 0.528, 0.573, 0.516, 0.545])
mps_challenge['rsa_2'] = np.array([0.300, 0.286, 0.305, 0.279, 0.288])
css_challenge['rsa_2'] = np.array([0.363, 0.358, 0.402, 0.362, 0.308])

means_challenge['ibr_2'] = np.array([0.784, 0.800, 0.840, 0.835, 0.770])
mps_challenge['ibr_2'] = np.array([0.363, 0.342, 0.360, 0.332, 0.340])
css_challenge['ibr_2'] = np.array([0.609, 0.620, 0.670, 0.656, 0.587])

means_challenge['rsa_3'] = np.array([0.523, 0.523, 0.569, 0.506, 0.555])
mps_challenge['rsa_3'] = np.array([0.292, 0.280, 0.301, 0.273, 0.292])
css_challenge['rsa_3'] = np.array([0.350, 0.360, 0.404, 0.355, 0.318])

means_challenge['ibr_3'] = np.array([0.784, 0.800, 0.840, 0.835, 0.770])
mps_challenge['ibr_3'] = np.array([0.363, 0.342, 0.360, 0.332, 0.340])
css_challenge['ibr_3'] = np.array([0.609, 0.620, 0.670, 0.656, 0.587])

means_challenge['rsa_equi'] = np.array([0.553, 0.518, 0.564, 0.538, 0.580])
mps_challenge['rsa_equi'] = np.array([0.309, 0.277, 0.298, 0.293, 0.302])
css_challenge['rsa_equi'] = np.array([0.378, 0.352, 0.389, 0.376, 0.333])

means_challenge['rsa_equi_virtual'] = np.array([0.518, 0.559, 0.551, 0.538, 0.550])
mps_challenge['rsa_equi_virtual'] = np.array([0.293, 0.312, 0.308, 0.308, 0.300])
css_challenge['rsa_equi_virtual'] = np.array([0.355, 0.392, 0.409, 0.380, 0.355])

means_challenge['ibr_equi'] = np.array([0.784, 0.800, 0.840, 0.835, 0.770])
mps_challenge['ibr_equi'] = np.array([0.363, 0.342, 0.360, 0.332, 0.340])
css_challenge['ibr_equi'] = np.array([0.609, 0.620, 0.670, 0.656, 0.587])

means_challenge['ibr_equi_virtual'] = np.array([0.698, 0.656, 0.702, 0.687, 0.685])
mps_challenge['ibr_equi_virtual'] = np.array([0.339, 0.296, 0.323, 0.298, 0.312])
css_challenge['ibr_equi_virtual'] = np.array([0.553, 0.518, 0.561, 0.554, 0.527])

means_challenge['table'] = np.array([0.729, 0.733, 0.778, 0.769, 0.765])
mps_challenge['table'] = np.array([0.343, 0.330, 0.343, 0.342, 0.334])
css_challenge['table'] = np.array([0.515, 0.516, 0.594, 0.569, 0.517])

means_challenge['table_virtual'] = np.array([0.603, 0.549, 0.587, 0.604, 0.560])
mps_challenge['table_virtual'] = np.array([0.285, 0.265, 0.289, 0.274, 0.266])
css_challenge['table_virtual'] = np.array([0.434, 0.415, 0.456, 0.449, 0.388])

means_challenge['table_special'] = np.array([0.940, 0.949, 0.933, 0.945, 0.935])
mps_challenge['table_special'] = np.array([0.349, 0.324, 0.342, 0.346, 0.328])
css_challenge['table_special'] = np.array([0.613, 0.635, 0.664, 0.665, 0.601])

means_challenge['table_special_virtual'] = np.array([0.698, 0.667, 0.653, 0.698, 0.680])
mps_challenge['table_special_virtual'] = np.array([0.243, 0.236, 0.255, 0.231, 0.232])
css_challenge['table_special_virtual'] = np.array([0.474, 0.475, 0.489, 0.499, 0.438])

means_challenge[''] = np.array([])
mps_challenge[''] = np.array([])
css_challenge[''] = np.array([])


nepoch = 5
ce = st.t.ppf((1 + 0.95) / 2., nepoch-1)

for key in keys:
    mps_challenge_real[key] =  mps_challenge[key] / means_challenge[key]
    css_challenge_real[key] =  css_challenge[key] / means_challenge[key]

    print('\n'+key+' challenge')
    print('reward %.1f %.1f %.0f'%(np.mean(means_challenge[key])*100, np.std(means_challenge[key])*100, st.sem(means_challenge[key]) * ce))
    print('message_prob %.2f %.2f %.0f'%(np.mean(mps_challenge_real[key]), np.std(mps_challenge_real[key]), st.sem(mps_challenge_real[key]) * ce))
    print('choice_prob %.2f %.2f %.0f'%(np.mean(css_challenge_real[key]), np.std(css_challenge_real[key]), st.sem(css_challenge_real[key]) * ce))