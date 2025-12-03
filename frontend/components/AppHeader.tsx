import React from 'react';
import { View, Image } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';

const AppHeader = () => {

    return (
        <View>
            <LinearGradient
                colors={['#FF6B00', '#FFAB7B']}
                className="items-center py-2"
            >
                <Image
                    source={require('../assets/images/GesturiX-word.png')}
                    className="w-40 h-10"
                />
            </LinearGradient>

            <LinearGradient
                colors={[
                    'rgba(255, 171, 123, 1.0)', 
                    'rgba(255, 171, 123, 0.0)' 
                ]} 
                start={{ x: 0.5, y: 0.0 }} 
                end={{ x: 0.5, y: 1.0 }} 
                className="items-center py-2"
            />
        </View>
    );
};

export default AppHeader;