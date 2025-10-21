/** @type {import('tailwindcss').Config} */
module.exports = {
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
        darkbg: '#1A1A1A',
        darksurface: '#333333',
        darkhover: '#444444',
        lighthover: '#EBEBEB',
      },

      // 🎨 FONT SETUP
      fontFamily: {
        // 🧠 Existing fonts (Inter, Montserrat, Orbitron)
        'inter': ['Inter-Regular'],
        'inter-bold': ['Inter-Bold'],
        'inter-italic': ['Inter-Italic'],
        'inter-bolditalic': ['Inter-BoldItalic'],

        'montserrat': ['Montserrat-Regular'],
        'montserrat-semibold': ['Montserrat-SemiBold'],
        'montserrat-bold': ['Montserrat-Bold'],
        'montserrat-italic': ['Montserrat-Italic'],
        'montserrat-bolditalic': ['Montserrat-BoldItalic'],

        'orbitron': ['Orbitron-Regular'],
        'orbitron-bold': ['Orbitron-Bold'],

        'fredoka': ['Fredoka-Regular'],
        'fredoka-medium': ['Fredoka-Medium'],
        'fredoka-semibold': ['Fredoka-SemiBold'],
        'fredoka-bold': ['Fredoka-Bold'],

        'audiowide': ['Audiowide-Regular'],
      },
    },
  },
  plugins: [],
  darkMode: 'class',
}