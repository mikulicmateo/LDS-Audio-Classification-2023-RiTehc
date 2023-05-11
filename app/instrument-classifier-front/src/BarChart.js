import React from "react";
import { Bar } from "react-chartjs-2";
import { Chart as ChartJS } from "chart.js/auto";

const options = {
  responsive: true,
  scales: {
    x: {
      grid: {
        color: 'rgba(85, 60, 154, 0.7)',
        display: false
      },
      ticks: {
        color: "rgba(85, 60, 154, 1)"
      },
    },
    y: {
      grid: {
        color: 'rgba(85, 60, 154, 0.7)'
      },
      beginAtZero: true,
      ticks: {
        color: "rgba(85, 60, 154, 1)"
      },
      max: 1
    }
  }
}

function BarChart({ chartData }) {
  return <Bar data={chartData} options={options}/>;
}

export default BarChart;