import React from 'react';
import { View, Text, TextInput, TouchableOpacity, ScrollView, Image } from 'react-native';
import { MaterialIcons, MaterialCommunityIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import AppHeader from '../../components/AppHeader'; 

const Compose = () => {
    const insets = useSafeAreaInsets();
    return (
        <View className="flex-1 bg-secondary" style={{ paddingTop: insets.top }}>
            
            <AppHeader /> 

            <ScrollView className="flex-1 p-4">
                {/* Text Input Section */}
                <View className="bg-white rounded-xl shadow-md p-4 mb-6">
                    <View className="flex-row justify-between items-center mb-4">
                        <Text className="text-base font-semibold text-primary">Type or Speak</Text>
                        <TouchableOpacity className="bg-accent rounded-full p-2 shadow-sm">
                            <MaterialCommunityIcons name="microphone" size={20} color="white" />
                        </TouchableOpacity>
                    </View>
                    <TextInput
                        className="h-24 p-4 text-base bg-secondary rounded-lg border border-neutral/30 text-primary"
                        placeholder="Type something to see its sign..."
                        multiline={true}
                    />
                </View>

                {/* Sign Language Video Section */}
                <Text className="text-base font-semibold text-primary mb-2">Sign Language Video</Text>
                <View className="w-full aspect-[4/3] bg-primary rounded-2xl overflow-hidden mb-5 relative">
                    {/* Placeholder for video feed */}
                    <View className="flex-1 justify-center items-center px-5">
                        <MaterialIcons 
                            name="videocam" 
                            size={80} 
                            color="#A8A8A8" 
                        />
                        <Text className="text-lg font-semibold text-neutral mt-4 mb-2">
                            Video Placeholder
                        </Text>
                        <Text className="text-sm text-neutral/50 text-center leading-5">
                            The corresponding sign will appear here
                        </Text>
                    </View>
                    
                    {/* Video overlay corners */}
                    <View className="absolute top-4 left-4 w-6 h-6 border-t-[3px] border-l-[3px] border-accent" />
                    <View className="absolute top-4 right-4 w-6 h-6 border-t-[3px] border-r-[3px] border-accent" />
                    <View className="absolute bottom-4 left-4 w-6 h-6 border-b-[3px] border-l-[3px] border-accent" />
                    <View className="absolute bottom-4 right-4 w-6 h-6 border-b-[3px] border-r-[3px] border-accent" />
                </View>

                {/* Quick Phrases Section */}
                <Text className="text-base font-semibold text-primary mb-2">Quick Phrases</Text>
                <View className="flex-row flex-wrap justify-between">
                    <TouchableOpacity className="bg-highlight rounded-full px-4 py-2 my-1 shadow-sm">
                        <Text className="text-primary text-sm font-semibold">Hello</Text>
                    </TouchableOpacity>
                    <TouchableOpacity className="bg-highlight rounded-full px-4 py-2 my-1 shadow-sm">
                        <Text className="text-primary text-sm font-semibold">Thank You</Text>
                    </TouchableOpacity>
                    <TouchableOpacity className="bg-highlight rounded-full px-4 py-2 my-1 shadow-sm">
                        <Text className="text-primary text-sm font-semibold">How are you?</Text>
                    </TouchableOpacity>
                    <TouchableOpacity className="bg-highlight rounded-full px-4 py-2 my-1 shadow-sm">
                        <Text className="text-primary text-sm font-semibold">I love you</Text>
                    </TouchableOpacity>
                    <TouchableOpacity className="bg-highlight rounded-full px-4 py-2 my-1 shadow-sm">
                        <Text className="text-primary text-sm font-semibold">Good morning</Text>
                    </TouchableOpacity>
                </View>
            </ScrollView>
        </View>
    )
}

export default Compose;