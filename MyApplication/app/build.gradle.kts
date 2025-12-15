plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.example.myapplication"
    compileSdk = 34 // Target Android 14

    defaultConfig {
        minSdk = 26
        targetSdk = 34
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

// ===================================================
// ADD THIS REPOSITORIES BLOCK HERE
// ===================================================
repositories {
    google()
    mavenCentral()
}

dependencies {
    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.activity)
    implementation(libs.constraintlayout)

    testImplementation(libs.junit)
    androidTestImplementation(libs.ext.junit)
    androidTestImplementation(libs.espresso.core)

    // FIX 1: TENSORFLOW LITE IMPLEMENTATION (EXCLUDE API)
    // This resolves the "Duplicate class org.tensorflow.lite.DataType" error
    // by stopping the primary TFLite library from importing the API classes.
    implementation("org.tensorflow:tensorflow-lite:2.17.0") {
        exclude(group = "org.tensorflow", module = "tensorflow-lite-api")
    }

    // FIX 2: TENSORFLOW LITE SUPPORT IMPLEMENTATION (EXCLUDE API)
    // The support library might also be pulling in the duplicate API classes.
    implementation("org.tensorflow:tensorflow-lite-support:0.4.4") {
        exclude(group = "org.tensorflow", module = "tensorflow-lite-api")
    }

    // Other dependencies
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
}