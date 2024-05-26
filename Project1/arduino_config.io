const socket = new WebSocket('ws://192.168.1.177:81'); // Replace with your Arduino's IP address

socket.onopen = function(event) {
  console.log('WebSocket connection established');
};

const luxChart = new Chart(document.getElementById('luxChart').getContext('2d'), {
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      label: 'LUX Value',
      data: [],
      backgroundColor: 'rgba(255, 99, 132, 0.2)',
      borderColor: 'rgba(255, 99, 132, 1)',
      borderWidth: 1
    }]
  },
  options: {
    scales: {
      y: {
        beginAtZero: true
      }
    }
  }
});

socket.onmessage = function(event) {
  const luxValue = event.data;
  const time = new Date().toLocaleTimeString();
  luxChart.data.labels.push(time);
  luxChart.data.datasets[0].data.push(luxValue);
  luxChart.update();
};
