// File: ../../components/AppHeader.tsx

import React from 'react';
import { View, Image } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';

// Define the component properties if needed, but for a static logo, none are required.
const AppHeader = () => {

    return (
        // Use a View to apply paddingTop from insets, making sure content is below the status bar
        <View>
            <LinearGradient
                colors={['#FF6B00', '#FFAB7B']} // Original gradient colors
                className="items-center py-2" // Slightly reduced padding for a cleaner look
            >
                {/* Replace the Text with Image
                    🚨 IMPORTANT: Update the 'source' path to your actual logo image. 
                */}
                <Image
                    source={require('../assets/images/GesturiX-word.png')} // 👈 UPDATE THIS PATH
                    className="w-40 h-10" // Adjust size as needed (e.g., w-40 h-10 for a wide, short logo)
                />
            </LinearGradient>

            <LinearGradient
                // The colors array defines the gradient stops:
                colors={['#FFAB7B', '#FFAB7B00']}
                className="items-center py-2"
            />
        </View>
    );
};

export default AppHeader;