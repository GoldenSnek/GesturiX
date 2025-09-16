import React, { useState } from 'react';
import { 
    View, 
    Text, 
    TouchableOpacity, 
    StatusBar,
} from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';

export default function Translate() {
    const [isTranslating, setIsTranslating] = useState(false);
    const [translatedText, setTranslatedText] = useState('Start signing to see translation...');
    const insets = useSafeAreaInsets();

    const toggleTranslation = () => {
        setIsTranslating(!isTranslating);
        if (!isTranslating) {
            setTranslatedText('Recognizing signs...');
            // Placeholder for translation delay
            setTimeout(() => {
                setTranslatedText('Hello, how are you today?');
            }, 2000);
        } else {
            setTranslatedText('Start signing to see translation...');
        }
    };

    return (
        <View className="flex-1 bg-secondary" style={{ paddingTop: insets.top }}>
            <StatusBar barStyle="light-content" />
            
            {/* Header */}
            <LinearGradient
                colors={['#FF6B00', '#FFAB7B']}
                className="items-center py-5"
            >
                <Text className="text-primary text-3xl font-bold">GesturiX</Text>
            </LinearGradient>

            {/* Camera Viewfinder & Control Button */}
            <View className="flex-1 px-5 pt-5 items-center relative">
                <View className="w-full aspect-[4/3] bg-primary rounded-2xl overflow-hidden mb-5 relative">
                    {/* Placeholder for camera feed */}
                    <View className="flex-1 justify-center items-center px-5">
                        <MaterialIcons 
                            name="video-camera-front" 
                            size={80} 
                            color="#A8A8A8" 
                        />
                        <Text className="text-lg font-semibold text-neutral mt-4 mb-2">
                            Camera View
                        </Text>
                        <Text className="text-sm text-neutral/50 text-center leading-5">
                            Position yourself in frame and start signing
                        </Text>
                    </View>
                    
                    {/* Camera overlay corners */}
                    <View className="absolute top-4 left-4 w-6 h-6 border-t-[3px] border-l-[3px] border-accent" />
                    <View className="absolute top-4 right-4 w-6 h-6 border-t-[3px] border-r-[3px] border-accent" />
                    <View className="absolute bottom-4 left-4 w-6 h-6 border-b-[3px] border-l-[3px] border-accent" />
                    <View className="absolute bottom-4 right-4 w-6 h-6 border-b-[3px] border-r-[3px] border-accent" />
                </View>

                {/* Control Button */}
                <TouchableOpacity 
                    className={`
                        w-[70px] h-[70px] rounded-full justify-center items-center
                        shadow-lg shadow-black/30
                        ${isTranslating ? 'bg-highlight' : 'bg-accent'}
                    `}
                    onPress={toggleTranslation}
                >
                    <MaterialIcons 
                        name={isTranslating ? 'stop' : 'play-arrow'} 
                        size={32} 
                        color="white" 
                    />
                </TouchableOpacity>
            </View>

            {/* Translation Output */}
            <View className="px-5 pb-5">
                <Text className="text-base font-semibold text-primary mb-3">Translation:</Text>
                <View className="bg-white rounded-xl p-5 min-h-[80px] border border-neutral/20 shadow-sm">
                    <Text className="text-lg text-primary leading-6 text-center">
                        {translatedText}
                    </Text>
                </View>
                
                {/* Status Indicator */}
                <View className="flex-row items-center justify-center mt-3">
                    <View 
                        className={`w-2 h-2 rounded-full mr-2 
                            ${isTranslating ? 'bg-accent' : 'bg-neutral'}
                        `} 
                    />
                    <Text className="text-sm text-neutral font-medium">
                        {isTranslating ? 'Translating...' : 'Ready'}
                    </Text>
                </View>
            </View>
        </View>
    );
}