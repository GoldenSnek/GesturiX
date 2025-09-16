/** @type {import('tailwindcss').Config} */
module.exports = {
  // NOTE: Update this to include the paths to all files that contain Nativewind classes.
  content: ["./App.tsx", "./app/**/*.{js,jsx,ts,tsx}", "./components/**/*.{js,jsx,ts,tsx}"],
  presets: [require("nativewind/preset")],
  theme: {
    extend: {
      colors: {
        primary: '#2C2C2C',
        secondary: '#F8F8F8',
        accent: '#FF6B00',
        highlight: '#FFAB7B',
        neutral: '#A8A8A8',
      }
    },
  },
  plugins: [],
}