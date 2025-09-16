import React, { useState } from 'react';
import { View, Text, Switch, TouchableOpacity, ScrollView, Image } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { MaterialIcons } from '@expo/vector-icons';

const Settings = () => {
    const insets = useSafeAreaInsets();

    const [isAudioOutputEnabled, setAudioOutputEnabled] = useState(false);
    const [isHapticFeedbackEnabled, setHapticFeedbackEnabled] = useState(false);
    const [isAudioRecordEnabled, setAudioRecordEnabled] = useState(false);
    const [isDarkModeEnabled, setDarkModeEnabled] = useState(false);
    const [areNotificationsEnabled, setNotificationsEnabled] = useState(false);

    const toggleAudioOutput = () => setAudioOutputEnabled(previousState => !previousState);
    const toggleHapticFeedback = () => setHapticFeedbackEnabled(previousState => !previousState);
    const toggleAudioRecord = () => setAudioRecordEnabled(previousState => !previousState);
    const toggleDarkMode = () => setDarkModeEnabled(previousState => !previousState);
    const toggleNotifications = () => setNotificationsEnabled(previousState => !previousState);

    return (
        <View className="flex-1 bg-secondary" style={{ paddingTop: insets.top }}>
            {/* Header */}
            <LinearGradient
                colors={['#FF6B00', '#FFAB7B']}
                className="items-center py-5"
            >
                <Text className="text-primary text-3xl font-bold">GesturiX</Text>
            </LinearGradient>

            <ScrollView className="flex-1 p-4" style={{ paddingBottom: 100 }}>
                {/* Translation Section */}
                <Text className="text-lg font-semibold text-primary mb-4">Translation</Text>
                <View className="bg-white rounded-xl p-4 shadow-md mb-6">
                    {/* Sign Language Option */}
                    <View className="flex-row items-center justify-between mb-4">
                        <View className="flex-row items-center">
                            <MaterialIcons name="language" size={24} color="#555" />
                            <View className="ml-4">
                                <Text className="text-base font-medium text-primary">Sign Language</Text>
                                <Text className="text-sm text-neutral">Currently: American Sign Language (ASL)</Text>
                            </View>
                        </View>
                        <TouchableOpacity className="flex-row items-center bg-secondary rounded-full px-4 py-2">
                            <Text className="text-sm font-semibold text-primary mr-2">ASL</Text>
                            <MaterialIcons name="chevron-right" size={20} color="#6B7280" />
                        </TouchableOpacity>
                    </View>

                    {/* Audio Output Toggle */}
                    <View className="flex-row items-center justify-between py-2">
                        <View className="flex-row items-center">
                            <MaterialIcons name="volume-up" size={24} color="#555" />
                            <View className="ml-4">
                                <Text className="text-base font-medium text-primary">Audio Output</Text>
                                <Text className="text-sm text-neutral">Speak translated text aloud</Text>
                            </View>
                        </View>
                        <Switch
                            trackColor={{ false: "#E5E7EB", true: "#FF6B00" }}
                            thumbColor={isAudioOutputEnabled ? "#fff" : "#f4f3f4"}
                            ios_backgroundColor="#E5E7EB"
                            onValueChange={toggleAudioOutput}
                            value={isAudioOutputEnabled}
                        />
                    </View>
                    
                    {/* Haptic Feedback Toggle */}
                    <View className="flex-row items-center justify-between py-2">
                        <View className="flex-row items-center">
                            <MaterialIcons name="vibration" size={24} color="#555" />
                            <View className="ml-4">
                                <Text className="text-base font-medium text-primary">Haptic Feedback</Text>
                                <Text className="text-sm text-neutral">Vibrate on successful recognition</Text>
                            </View>
                        </View>
                        <Switch
                            trackColor={{ false: "#E5E7EB", true: "#FF6B00" }}
                            thumbColor={isHapticFeedbackEnabled ? "#fff" : "#f4f3f4"}
                            ios_backgroundColor="#E5E7EB"
                            onValueChange={toggleHapticFeedback}
                            value={isHapticFeedbackEnabled}
                        />
                    </View>
                </View>

                {/* Camera Section */}
                <Text className="text-lg font-semibold text-primary mb-4">Camera</Text>
                <View className="bg-white rounded-xl p-4 shadow-md mb-6">
                    {/* Audio Record Toggle */}
                    <View className="flex-row items-center justify-between py-2">
                        <View className="flex-row items-center">
                            <MaterialIcons name="mic-none" size={24} color="#555" />
                            <View className="ml-4">
                                <Text className="text-base font-medium text-primary">Audio Record</Text>
                                <Text className="text-sm text-neutral">Start recording when hands detected</Text>
                            </View>
                        </View>
                        <Switch
                            trackColor={{ false: "#E5E7EB", true: "#FF6B00" }}
                            thumbColor={isAudioRecordEnabled ? "#fff" : "#f4f3f4"}
                            ios_backgroundColor="#E5E7EB"
                            onValueChange={toggleAudioRecord}
                            value={isAudioRecordEnabled}
                        />
                    </View>
                </View>

                {/* Display Section */}
                <Text className="text-lg font-semibold text-primary mb-4">Display</Text>
                <View className="bg-white rounded-xl p-4 shadow-md mb-6">
                    <View className="flex-row items-center justify-between py-2">
                        <View className="flex-row items-center">
                            <MaterialIcons name="dark-mode" size={24} color="#555" />
                            <View className="ml-4">
                                <Text className="text-base font-medium text-primary">Dark Mode</Text>
                                <Text className="text-sm text-neutral">Switch to a darker theme</Text>
                            </View>
                        </View>
                        <Switch
                            trackColor={{ false: "#E5E7EB", true: "#FF6B00" }}
                            thumbColor={isDarkModeEnabled ? "#fff" : "#f4f3f4"}
                            ios_backgroundColor="#E5E7EB"
                            onValueChange={toggleDarkMode}
                            value={isDarkModeEnabled}
                        />
                    </View>
                </View>

                {/* Help & Support Section */}
                <Text className="text-lg font-semibold text-primary mb-4">Help & Support</Text>
                <View className="bg-white rounded-xl p-4 shadow-md mb-6">
                    <TouchableOpacity className="flex-row items-center py-2">
                        <MaterialIcons name="help-outline" size={24} color="#555" />
                        <View className="ml-4">
                            <Text className="text-base font-medium text-primary">FAQ</Text>
                        </View>
                    </TouchableOpacity>
                    <View className="border-b border-gray-200 my-2" />
                    <TouchableOpacity className="flex-row items-center py-2">
                        <MaterialIcons name="mail-outline" size={24} color="#555" />
                        <View className="ml-4">
                            <Text className="text-base font-medium text-primary">Contact Us</Text>
                        </View>
                    </TouchableOpacity>
                    <View className="border-b border-gray-200 my-2" />
                    <TouchableOpacity className="flex-row items-center py-2">
                        <MaterialIcons name="lock-outline" size={24} color="#555" />
                        <View className="ml-4">
                            <Text className="text-base font-medium text-primary">Privacy Policy</Text>
                        </View>
                    </TouchableOpacity>
                </View>

                {/* About Section */}
                <Text className="text-lg font-semibold text-primary mb-4">About</Text>
                <View className="bg-white rounded-xl p-4 shadow-md flex-row items-start mb-6">
                    <Image
                        source={require('../../assets/images/GesturiX.png')}
                        className="w-16 h-16 mr-4 rounded-xl"
                    />
                    <View className="flex-1">
                        <Text className="text-lg font-bold text-primary">GesturiX</Text>
                        <Text className="text-sm text-neutral mb-2">Version 1.0.0</Text>
                        <Text className="text-sm text-neutral leading-5">
                            GesturiX is a mobile app that bridges communication between the deaf and hard-of-hearing community and the general public. It empowers users with real-time sign language recognition, translating gestures into text or speech and vice versa. As a school project, it showcases the practical use of computer vision and machine learning in creating a socially impactful solution.
                        </Text>
                    </View>
                </View>
            </ScrollView>
        </View>
    );
}

export default Settings;