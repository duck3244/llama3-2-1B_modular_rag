/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: {
          50: '#f7f7f9',
          100: '#eceef2',
          800: '#1f2330',
          900: '#13151c',
        },
      },
    },
  },
  plugins: [],
};
